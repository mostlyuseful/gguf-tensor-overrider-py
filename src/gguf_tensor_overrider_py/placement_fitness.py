import re
from typing import List, Dict, Any, Optional, Set

# --- Heuristic Penalty Constants ---
# Why these values? The absolute numbers are less important than their relative
# magnitude. They establish a clear hierarchy of "badness".
# A CPU tensor is orders of magnitude worse than a layer transition.
PENALTY_CPU_TENSOR = (
    1_000_000.0  # For any tensor on the main compute path placed on CPU.
)
PENALTY_INTRA_OP_SPLIT = (
    100_000.0  # For splitting a single operation (e.g., weight/bias).
)
PENALTY_MOE_ROUTER_MISMATCH = 5_000.0  # For router on different GPU than its experts.
PENALTY_SCATTERED_EXPERTS = 2_000.0  # Per extra GPU used for one layer's experts.
PENALTY_LAYER_TRANSITION = (
    500.0  # For hidden state transfer between layers on different GPUs.
)


class LLMConfigMoE:
    """Expanded LLMConfig to include MoE parameters for fitness evaluation."""

    def __init__(self, n_layers: int, n_experts: int = 0, n_experts_per_tok: int = 0):
        """
        Args:
            n_layers (int): The total number of decoder layers.
            n_experts (int): The number of experts in each MoE layer. 0 for dense models.
            n_experts_per_tok (int): The number of experts selected per token (e.g., top_k).
        """
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.n_experts_per_tok = n_experts_per_tok
        self.is_moe = n_experts > 0


def estimate_placement_fitness(
    placement_map: Dict[str, str], model_config: LLMConfigMoE
) -> float:
    """
    Evaluates a tensor placement strategy and returns a fitness score.
    A lower score indicates a better (higher performance) placement.

    Args:
        placement_map (Dict[str, str]): A map of tensor names to device strings
                                         (e.g., {'blk.0.attn_q.weight': 'CUDA0'}).
        model_config (LLMConfigMoE): The model's architectural configuration.

    Returns:
        float: The total penalty score. Lower is better.
    """
    total_penalty = 0.0

    # --- Step 1: Parse the flat map into a structured hierarchy ---
    # Why? Analyzing relationships (like weight/bias pairs or FFN components)
    # is impossible with a flat list. We need to structure the data by layer
    # and component to understand the data flow.
    layer_struct: Dict[int, Dict[str, Any]] = {
        i: {} for i in range(model_config.n_layers)
    }

    # Regex to capture layer, block, and tensor details.
    tensor_regex = re.compile(r"blk\.(\d+)\.(.*)")

    for name, device in placement_map.items():
        match = tensor_regex.match(name)
        if not match:
            continue  # Skip non-layer tensors like 'token_embd.weight' for now

        layer_idx, tensor_key = int(match.group(1)), match.group(2)

        # Deconstruct keys like 'ffn_expert0.gate.weight'
        keys = tensor_key.split(".")

        # Navigate/create path in the structured dict
        current_level = layer_struct[layer_idx]
        for key in keys[:-1]:
            if key not in current_level:
                current_level[key] = {}
            current_level = current_level[key]
        current_level[keys[-1]] = device

    # A helper to get all devices used within a dictionary sub-tree.
    def get_devices_in_block(block: Any) -> Set[str]:
        devices = set()
        if isinstance(block, dict):
            for value in block.values():
                devices.update(get_devices_in_block(value))
        elif isinstance(block, str):
            devices.add(block)
        return devices

    # --- Step 2: Apply penalties by analyzing the structured data ---

    # Track the output device of the previous layer to check for transitions.
    prev_layer_output_device: Optional[str] = "CUDA0"  # Assume input comes from GPU0

    for i in range(model_config.n_layers):
        layer = layer_struct[i]

        # --- A) Intra-Layer Penalties (High Severity) ---

        # Check all projections for splits (attn_q, k, v, o)
        for proj_name in ["attn_q", "attn_k", "attn_v", "attn_o"]:
            if proj_name in layer:
                devices = get_devices_in_block(layer[proj_name])
                if "CPU" in devices:
                    total_penalty += PENALTY_CPU_TENSOR
                if len(devices) > 1:
                    total_penalty += PENALTY_INTRA_OP_SPLIT

        # --- B) FFN/MoE Block Penalties ---
        layer_output_device = None

        if model_config.is_moe:
            # MoE Layer Logic
            moe_block = layer.get("ffn_moe", {})
            gate_block = layer.get("ffn_gate", {})  # Router

            gate_devices = get_devices_in_block(gate_block)
            if "CPU" in gate_devices:
                total_penalty += PENALTY_CPU_TENSOR
            if len(gate_devices) > 1:
                total_penalty += PENALTY_INTRA_OP_SPLIT

            expert_devices = set()
            for expert_idx in range(model_config.n_experts):
                expert_block = moe_block.get(f"expert{expert_idx}", {})
                devices = get_devices_in_block(expert_block)
                if "CPU" in devices:
                    total_penalty += PENALTY_CPU_TENSOR

                # Penalize splitting one expert's internal FFN.
                if len(devices) > 1:
                    total_penalty += PENALTY_INTRA_OP_SPLIT
                expert_devices.update(devices)

            # Penalize scattering experts across many GPUs.
            # Why? If top_k=2 and the chosen experts are on GPU0 and GPU1,
            # the hidden state must be sent to both, and results gathered.
            # This encourages grouping experts.
            num_gpus_for_experts = len([d for d in expert_devices if "CUDA" in d])
            if num_gpus_for_experts > 1:
                total_penalty += PENALTY_SCATTERED_EXPERTS * (num_gpus_for_experts - 1)

            # Penalize if router and experts are on different devices.
            if (
                gate_devices
                and expert_devices
                and not gate_devices.intersection(expert_devices)
            ):
                total_penalty += PENALTY_MOE_ROUTER_MISMATCH

            # The output of the layer comes from where the experts are.
            layer_output_device = list(expert_devices)[0] if expert_devices else None

        else:
            # Dense FFN Layer Logic
            ffn_block = layer.get("ffn", {})
            devices = get_devices_in_block(ffn_block)
            if "CPU" in devices:
                total_penalty += PENALTY_CPU_TENSOR
            if len(devices) > 1:
                total_penalty += PENALTY_INTRA_OP_SPLIT
            layer_output_device = list(devices)[0] if devices else None

        # --- C) Inter-Layer Transition Penalty (Lower Severity) ---
        # Why? The entire hidden state must be copied if the next layer's
        # input block (attention) is on a different device than this layer's output.
        if (
            i > 0
            and layer_output_device
            and prev_layer_output_device != layer_output_device
        ):
            total_penalty += PENALTY_LAYER_TRANSITION

        if layer_output_device:
            prev_layer_output_device = layer_output_device

    return total_penalty


# ==========================================================
# ---                  EXAMPLE USAGE                     ---
# ==========================================================
if __name__ == "__main__":
    # Use a small MoE model for a comprehensive example
    mixtral_like_config = LLMConfigMoE(n_layers=4, n_experts=4, n_experts_per_tok=2)

    # --- Define Placement Strategies ---

    # Strategy A (Good): Clean split. Layers 0-1 on GPU0, 2-3 on GPU1. Experts are grouped.
    strategy_A = {}
    for i in range(2):
        strategy_A[f"blk.{i}.*"] = "CUDA0"
    for i in range(2, 4):
        strategy_A[f"blk.{i}.*"] = "CUDA1"

    # Strategy B (Okay but inefficient): Alternating layers.
    strategy_B = {}
    for i in range(4):
        strategy_B[f"blk.{i}.*"] = f"CUDA{i % 2}"

    # Strategy C (Bad): Split an operation within a layer.
    strategy_C = strategy_A.copy()
    strategy_C["blk.0.attn_q.weight"] = "CUDA0"
    strategy_C["blk.0.attn_q.bias"] = "CUDA1"  # The cardinal sin

    # Streategy D (Bad MoE): Scatter experts for a single layer across GPUs.
    strategy_D = strategy_A.copy()
    strategy_D["blk.0.ffn_moe.expert0.*"] = "CUDA0"
    strategy_D["blk.0.ffn_moe.expert1.*"] = "CUDA1"
    strategy_D["blk.0.ffn_moe.expert2.*"] = "CUDA0"
    strategy_D["blk.0.ffn_moe.expert3.*"] = "CUDA1"

    # --- Generate Flat Maps from Regex Rules (Simple Simulator) ---
    def generate_map(rules: Dict[str, str], config: LLMConfigMoE) -> Dict[str, str]:
        # This is a simplified version of llama.cpp's tensor resolution
        tensor_names = []
        for i in range(config.n_layers):
            for proj in ["attn_q", "attn_k", "attn_v", "attn_o"]:
                tensor_names.extend([f"blk.{i}.{proj}.weight", f"blk.{i}.{proj}.bias"])
            if config.is_moe:
                for exp_i in range(config.n_experts):
                    tensor_names.append(f"blk.{i}.ffn_moe.expert{exp_i}.gate.weight")
            else:  # dense
                tensor_names.append(f"blk.{i}.ffn.gate.weight")

        flat_map = {}
        for name in tensor_names:
            for pattern, device in reversed(list(rules.items())):
                if re.match(pattern.replace("*", ".*"), name):
                    flat_map[name] = device
                    break
        return flat_map

    # --- Evaluate and Compare ---
    map_A = generate_map(strategy_A, mixtral_like_config)
    map_B = generate_map(strategy_B, mixtral_like_config)
    map_C = generate_map(strategy_C, mixtral_like_config)
    map_D = generate_map(strategy_D, mixtral_like_config)

    score_A = estimate_placement_fitness(map_A, mixtral_like_config)
    score_B = estimate_placement_fitness(map_B, mixtral_like_config)
    score_C = estimate_placement_fitness(map_C, mixtral_like_config)
    score_D = estimate_placement_fitness(map_D, mixtral_like_config)

    print(
        f"Strategy A (Clean Split)    | Score: {score_A:12,.1f} | Interpretation: Good. Penalty is from one Layer 1 -> 2 transition."
    )
    print(
        f"Strategy B (Alternating)    | Score: {score_B:12,.1f} | Interpretation: Okay. Penalized for multiple layer transitions (0->1, 1->2, 2->3)."
    )
    print(
        f"Strategy D (Scattered Exp)  | Score: {score_D:12,.1f} | Interpretation: Bad. Higher penalty from scattered experts in layer 0."
    )
    print(
        f"Strategy C (Split Op)       | Score: {score_C:12,.1f} | Interpretation: Critical Error. Very high penalty from splitting a single operation."
    )
