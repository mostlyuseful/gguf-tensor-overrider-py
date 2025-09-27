"""Data models for GGUF tensor allocation system."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class TensorPriority(Enum):
    """Tensor allocation priority levels."""

    ATTENTION = 1
    FEED_FORWARD = 2
    GATE = 3
    NORM = 4
    OTHER = 5


class DataType(Enum):
    """Supported data types for KV cache and tensor calculations."""

    F16 = "f16"
    BF16 = "bf16"
    F32 = "f32"
    Q8_0 = "q8_0"
    Q4_0 = "q4_0"
    Q4_1 = "q4_1"
    IQ4_NL = "iq4_nl"
    Q5_0 = "q5_0"
    Q5_1 = "q5_1"

    @property
    def bytes_per_element(self) -> float:
        """Return bytes per element for this data type."""
        match self:
            case DataType.F16 | DataType.BF16:
                return 2
            case DataType.F32:
                return 4
            case DataType.Q8_0:
                return 1
            # The following values are aligned to weight-quant style approximations.
            # KV cache uses different packing; KV-specific mapping is handled in KVCacheConfig.
            case DataType.Q4_0:
                return 4.0
            case DataType.Q4_1:
                return 4.5
            case DataType.IQ4_NL:
                return 4.0
            case DataType.Q5_0:
                return 5.5
            case DataType.Q5_1:
                return 5.0
            case _:
                raise ValueError(f"Unknown data type: {self.value}")


@dataclass
class TensorInfo:
    """Information about a single tensor from GGUF file."""

    name: str
    size_bytes: int
    block_id: Optional[int] = None
    priority: TensorPriority = TensorPriority.OTHER

    def __post_init__(self) -> None:
        """Classify tensor and extract block ID after creation."""
        self.block_id = self._extract_block_id()
        self.priority = self._classify_priority()

    def _extract_block_id(self) -> Optional[int]:
        """Extract block/layer index from tensor name using universal regex."""
        # Universal pattern: find first number in tensor name
        pattern = r"(?:^|[^0-9])(\d+)(?:\.|$)"
        match = re.search(pattern, self.name)
        return int(match.group(1)) if match else None

    def _classify_priority(self) -> TensorPriority:
        """Classify tensor by priority based on name patterns."""
        name_lower = self.name.lower()

        # Check for norm first (since layernorm contains both 'attention' and 'norm')
        if "norm" in name_lower:
            return TensorPriority.NORM

        # Attention tensors (highest priority)
        if "attention" in name_lower or "attn" in name_lower:
            return TensorPriority.ATTENTION

        # Feed-forward tensors (exclude expert/gate/norm)
        if "ffn" in name_lower or "feed_forward" in name_lower:
            if not any(
                excl in name_lower for excl in ["exp", "expert", "gate", "norm"]
            ):
                return TensorPriority.FEED_FORWARD

        # Gate tensors
        if "gate" in name_lower:
            return TensorPriority.GATE

        # Everything else
        return TensorPriority.OTHER


@dataclass
class GPUCapacity:
    """GPU memory capacity and allocation tracking."""

    gpu_id: int
    total_vram_bytes: int
    usable_percentage: float
    allocated_bytes: int = 0
    kv_cache_reserved_bytes: int = 0
    allocated_tensors: List[str] = field(default_factory=list)
    allocated_blocks: Set[Optional[int]] = field(default_factory=set)

    @property
    def usable_vram_bytes(self) -> int:
        """Total VRAM available for allocation (after percentage limit)."""
        return int(self.total_vram_bytes * self.usable_percentage / 100.0)

    @property
    def available_bytes(self) -> int:
        """Available VRAM for tensor allocation (after KV cache reservation)."""
        return (
            self.usable_vram_bytes - self.kv_cache_reserved_bytes - self.allocated_bytes
        )

    @property
    def utilization_percentage(self) -> float:
        """Current VRAM utilization as percentage of usable VRAM."""
        return (
            (self.allocated_bytes + self.kv_cache_reserved_bytes)
            / self.usable_vram_bytes
            * 100.0
        )

    def can_fit_tensor(self, size_bytes: int) -> bool:
        """Check if tensor of given size can fit in available space."""
        return self.available_bytes >= size_bytes
    
    def can_fit_tensor_group(self, sizes_bytes: List[int]) -> bool:
        """Check if a group of tensors can fit in available space."""
        total_size = sum(sizes_bytes)
        return self.available_bytes >= total_size

    def can_fit_block(self, block_tensors: List[TensorInfo]) -> bool:
        """Check if entire block can fit in available space."""
        total_size = sum(tensor.size_bytes for tensor in block_tensors)
        return self.available_bytes >= total_size

    def allocate_tensor(self, tensor: TensorInfo) -> None:
        """Allocate a tensor to this GPU."""
        if not self.can_fit_tensor(tensor.size_bytes):
            raise ValueError(
                f"Cannot fit tensor {tensor.name} ({tensor.size_bytes} bytes) "
                f"in GPU {self.gpu_id} (available: {self.available_bytes} bytes)"
            )

        self.allocated_bytes += tensor.size_bytes
        self.allocated_tensors.append(tensor.name)
        self.allocated_blocks.add(tensor.block_id)


@dataclass
class ModelMetadata:
    """Model architecture metadata extracted from GGUF."""

    architecture: str
    embedding_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: Optional[int] = None
    # Optional per-type head dimensions for architectures where K and V differ (e.g., MLA variants)
    k_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None

    def __post_init__(self) -> None:
        """Calculate head dimension if not provided."""
        if self.head_dim is None:
            if self.embedding_dim % self.n_heads != 0:
                raise ValueError(
                    f"Embedding dimension {self.embedding_dim} not divisible "
                    f"by number of heads {self.n_heads}"
                )
            self.head_dim = self.embedding_dim // self.n_heads


@dataclass
class KVCacheConfig:
    """KV cache configuration and memory requirements."""

    context_length: int
    k_dtype: DataType
    v_dtype: DataType

    def bytes_per_layer(self, metadata: ModelMetadata) -> int:
        """Calculate KV cache bytes per layer.

        Note: For MoE architectures (e.g., glm4moe), expert counts apply to
        feed-forward networks, not attention. KV cache stores K/V tensors for
        attention heads only and therefore is computed from attention-related
        dimensions (n_kv_heads and head_dim) regardless of expert_count.
        """

        # Map DataType to KV-cache bytes per element.
        def _kv_bytes(dt: DataType) -> float:
            # KV cache quantization in llama.cpp supports: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1.
            # Map to approximate bytes/element for KV tensors (ignoring small per-block scale overheads).
            if dt in (DataType.F16, DataType.BF16):
                return 2.0
            if dt is DataType.F32:
                return 4.0
            if dt is DataType.Q8_0:
                return 1.0
            if dt in (DataType.Q4_0, DataType.Q4_1, DataType.IQ4_NL):
                return 0.5
            if dt in (DataType.Q5_0, DataType.Q5_1):
                return 5.5 / 8  # 5.5 bits per value ≈ 0.6875 bytes
            # Fallback to enum's raw bytes (shouldn't happen)
            return float(dt.bytes_per_element)

        n_kv = metadata.n_kv_heads
        L = self.context_length

        # If architectures provide distinct key/value head dims, honor them
        if metadata.k_head_dim is not None and metadata.v_head_dim is not None:
            total = (
                L
                * n_kv
                * (
                    _kv_bytes(self.k_dtype) * metadata.k_head_dim
                    + _kv_bytes(self.v_dtype) * metadata.v_head_dim
                )
            )
            return int(math.ceil(total))

        # Fallback: use a single head_dim for both K and V
        if metadata.head_dim is None:
            raise ValueError("Head dimension not available for KV cache calculation")

        bytes_per_token = _kv_bytes(self.k_dtype) + _kv_bytes(self.v_dtype)
        total = L * n_kv * metadata.head_dim * bytes_per_token
        # Ensure integer bytes; ceil to avoid underestimation when sub-byte types are used
        return int(math.ceil(total))

    def total_bytes(self, metadata: ModelMetadata) -> int:
        """Calculate total KV cache bytes for entire model."""
        return self.bytes_per_layer(metadata) * metadata.n_layers


@dataclass
class BlockGroup:
    """Group of tensors belonging to the same model block/layer."""

    block_id: Optional[int]
    tensors: List[TensorInfo] = field(default_factory=list)

    @property
    def total_size_bytes(self) -> int:
        """Total size of all tensors in this block."""
        return sum(tensor.size_bytes for tensor in self.tensors)

    @property
    def is_global(self) -> bool:
        """Check if this is the global block (no specific layer ID)."""
        return self.block_id is None

    def add_tensor(self, tensor: TensorInfo) -> None:
        """Add a tensor to this block group."""
        if tensor.block_id != self.block_id:
            raise ValueError(
                f"Tensor {tensor.name} block ID {tensor.block_id} "
                f"doesn't match block group ID {self.block_id}"
            )
        self.tensors.append(tensor)

    def sort_by_priority(self) -> None:
        """Sort tensors within block by allocation priority."""
        self.tensors.sort(key=lambda t: t.priority.value)


@dataclass
class AllocationResult:
    """Mutable result of tensor allocation process."""

    gpu_allocations: Dict[int, GPUCapacity] = field(default_factory=dict)
    tensor_gpu_mapping: Dict[str, int] = field(default_factory=dict)
    unallocated_tensors: List[TensorInfo] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    _kv_config: Optional[KVCacheConfig] = field(default=None, init=False)
    _metadata: Optional[ModelMetadata] = field(default=None, init=False)

    def allocate_tensor_to_gpu(self, tensor: TensorInfo, gpu_id: int) -> None:
        """Allocate a tensor to specific GPU and update mappings."""
        if gpu_id not in self.gpu_allocations:
            raise ValueError(f"GPU {gpu_id} not found in available GPUs")

        gpu = self.gpu_allocations[gpu_id]
        gpu.allocate_tensor(tensor)
        self.tensor_gpu_mapping[tensor.name] = gpu_id

    def allocate_tensor_group_to_gpu(self, tensor_group: List[TensorInfo], gpu_id: int) -> None:
        """Allocate a group of tensors to specific GPU and update mappings."""
        if gpu_id not in self.gpu_allocations:
            raise ValueError(f"GPU {gpu_id} not found in available GPUs")

        gpu = self.gpu_allocations[gpu_id]
        tensor_sizes = [tensor.size_bytes for tensor in tensor_group]
        if not gpu.can_fit_tensor_group(tensor_sizes):
            raise ValueError(
                f"Cannot fit tensor group of size {sum(tensor_sizes)} bytes "
                f"in GPU {gpu_id} (available: {gpu.available_bytes} bytes)"
            )

        for tensor in tensor_group:
            gpu.allocate_tensor(tensor)
            self.tensor_gpu_mapping[tensor.name] = gpu_id

    def add_warning(self, message: str) -> None:
        """Add a warning message to the result."""
        self.warnings.append(message)

    def set_kv_config(self, kv_config: KVCacheConfig, metadata: ModelMetadata) -> None:
        """Set KV cache config and metadata for CPU calculations."""
        self._kv_config = kv_config
        self._metadata = metadata

    @property
    def total_allocated_bytes(self) -> int:
        """Total bytes allocated across all GPUs."""
        return sum(gpu.allocated_bytes for gpu in self.gpu_allocations.values())

    @property
    def total_kv_cache_bytes(self) -> int:
        """Total KV cache bytes reserved across all GPUs."""
        return sum(gpu.kv_cache_reserved_bytes for gpu in self.gpu_allocations.values())

    @property
    def cpu_kv_cache_bytes(self) -> int:
        """Calculate KV cache bytes that should be reserved on CPU for unallocated layers."""
        if not self._kv_config or not self._metadata or not self.unallocated_tensors:
            return 0

        # Find which layers have unallocated tensors
        unallocated_layers = set()
        for tensor in self.unallocated_tensors:
            if (
                tensor.block_id is not None
            ):  # Only count numbered layers, not global tensors
                unallocated_layers.add(tensor.block_id)

        # Calculate KV cache for unallocated layers
        if unallocated_layers:
            bytes_per_layer = self._kv_config.bytes_per_layer(self._metadata)
            return len(unallocated_layers) * bytes_per_layer

        return 0

    @property
    def total_cpu_bytes(self) -> int:
        """Total CPU memory required (unallocated tensors + CPU KV cache)."""
        unallocated_tensor_bytes = sum(t.size_bytes for t in self.unallocated_tensors)
        return unallocated_tensor_bytes + self.cpu_kv_cache_bytes

    @property
    def allocation_summary(self) -> Dict[str, Any]:
        """Generate allocation summary for output."""
        return {
            "total_tensors": len(self.tensor_gpu_mapping),
            "unallocated_tensors": len(self.unallocated_tensors),
            "total_allocated_bytes": self.total_allocated_bytes,
            "total_kv_cache_bytes": self.total_kv_cache_bytes,
            "gpu_utilization": {
                gpu_id: {
                    "allocated_bytes": gpu.allocated_bytes,
                    "kv_cache_bytes": gpu.kv_cache_reserved_bytes,
                    # Utilization relative to usable VRAM (historical behavior)
                    "utilization_percent": gpu.utilization_percentage,
                    # Additional context for clearer reporting
                    "total_vram_bytes": gpu.total_vram_bytes,
                    "usable_vram_bytes": gpu.usable_vram_bytes,
                    "usable_percent_limit": gpu.usable_percentage,
                    # Utilization relative to total VRAM (what users expect vs the max fill percent)
                    "utilization_percent_of_total": (
                        (gpu.allocated_bytes + gpu.kv_cache_reserved_bytes)
                        / gpu.total_vram_bytes
                        * 100.0
                        if gpu.total_vram_bytes > 0
                        else 0.0
                    ),
                    "allocated_blocks": sorted(
                        [b for b in gpu.allocated_blocks if b is not None]
                    ),
                    "tensor_count": len(gpu.allocated_tensors),
                }
                for gpu_id, gpu in self.gpu_allocations.items()
            },
            "warnings": self.warnings,
        }


@dataclass
class GPUConfiguration:
    """Configuration for GPU capacity (real or hypothetical)."""

    gpu_id: int
    total_vram_gb: float
    percentage: float = 90.0

    def to_gpu_capacity(self) -> GPUCapacity:
        """Convert to GPUCapacity for allocation tracking."""
        return GPUCapacity(
            gpu_id=self.gpu_id,
            total_vram_bytes=int(self.total_vram_gb * 1024**3),  # GB to bytes
            usable_percentage=self.percentage,
        )


@dataclass
class ArchitectureKeys:
    """Metadata key mappings for a specific architecture."""

    embedding_keys: List[str]
    layer_count_keys: List[str]
    n_heads_keys: List[str]
    n_kv_heads_keys: List[str]
    head_dim_keys: List[str]

    @classmethod
    def for_architecture(cls, arch: str) -> ArchitectureKeys:
        """Get key mappings for specific architecture."""
        return cls(
            embedding_keys=[f"{arch}.embedding_length", f"{arch}.hidden_size"],
            layer_count_keys=[
                f"{arch}.block_count",
                f"{arch}.n_layer",
                f"{arch}.layer_count",
            ],
            n_heads_keys=[f"{arch}.attention.head_count", f"{arch}.n_head"],
            n_kv_heads_keys=[
                f"{arch}.attention.head_count_kv",
                f"{arch}.n_kv_head",
                f"{arch}.rope.n_kv_head",
            ],
            # Many architectures expose head dimension using different names; include common aliases
            head_dim_keys=[
                f"{arch}.attention.head_dim",
                f"{arch}.head_dim",
                # Prefer value length when K/V differ, then key length
                f"{arch}.attention.value_length",
                f"{arch}.attention.key_length",
                # DeepSeek 2 MLA variants: prefer value then key
                f"{arch}.attention.value_length_mla",
                f"{arch}.attention.key_length_mla",
            ],
        )
