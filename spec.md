# GGUF Tensor Overrider Specification v1

## Overview

The GGUF Tensor Overrider is a Python tool that generates GPU allocation plans for llama.cpp and ik_llama.cpp tensor overrides. It parses GGUF model files to extract tensor information and metadata, then intelligently allocates tensors across available GPUs while reserving space for KV cache and maintaining optimal performance through block co-location.

## Core Requirements

### Primary Goals
- Parse GGUF files using the `gguf-parser` library to extract tensor info and metadata
- Generate GPU allocation plans that optimize VRAM usage across multiple GPUs
- Support both real GPU detection (via NVML) and hypothetical GPU configurations
- Maintain block co-location (same layer tensors on same GPU) for optimal performance
- Reserve appropriate VRAM for KV cache based on model architecture and context parameters

### Success Criteria
- Tool successfully parses GGUF files and extracts required metadata
- Allocation algorithm respects tensor priority, block co-location, and GPU capacity constraints
- Output format provides clear tensor→GPU mapping with utilization summary
- CLI interface matches the specification in README.md

## Tensor Classification and Priority

### Classification Rules (Priority Order)
Tensors are classified and allocated in the following priority order:

1. **Attention tensors** (highest priority)
   - Match: `'attention'` or `'attn'` in tensor names
   - Examples: `blk.0.attn_k.weight`, `layers.15.attention.q_proj.weight`

2. **Feed-forward tensors**
   - Match: `'ffn'` or `'feed_forward'` in tensor names
   - Exclude: tensors containing `'exp'`, `'expert'`, `'gate'`, or `'norm'`
   - Examples: `blk.0.ffn_down.weight`, `layers.15.feed_forward.intermediate.weight`

3. **Gate tensors**
   - Match: `'gate'` in tensor names
   - Examples: `blk.0.ffn_gate.weight`, `layers.15.mlp.gate_proj.weight`

4. **Norm tensors**
   - Match: `'norm'` in tensor names
   - Examples: `blk.0.attn_norm.weight`, `layers.15.input_layernorm.weight`

5. **Everything else** (lowest priority)
   - All remaining tensors not matching above categories
   - Examples: embeddings, output projections, etc.

### Block Co-location Requirement
- All tensors belonging to the same model layer/block MUST be allocated to the same GPU
- This ensures optimal performance by avoiding cross-GPU communication within a layer
- Block identification is done via regex pattern matching (see Block Grouping section)

## Architecture Metadata Mapping

### Architecture Detection
- Read `general.architecture` field from GGUF metadata (convert to lowercase)
- Support architectures with prefix matching: `[llama, qwen, qwen2, qwen2_moe, qwen3, qwen3moe, gemma3]`
- Unknown architectures should trigger a warning but allow fallback behavior

### Metadata Key Mapping (Tolerant Approach)
For each required parameter, try multiple possible keys in order until one is found:

#### Embedding Dimension
- Try: `["{arch}.embedding_length", "{arch}.hidden_size"]`
- Used for: KV cache sizing and head dimension inference

#### Layer Count  
- Try: `["{arch}.block_count", "{arch}.n_layer", "{arch}.layer_count"]`
- Used for: Total KV cache reservation and validation

#### Attention Head Count
- Try: `["{arch}.attention.head_count", "{arch}.n_head"]`
- Used for: Head dimension calculation and validation

#### KV Head Count (for Group Query Attention)
- Try: `["{arch}.attention.head_count_kv", "{arch}.n_kv_head", "{arch}.rope.n_kv_head"]`
- Used for: KV cache sizing (may be less than attention heads in GQA models)

#### Head Dimension
- Try: `["{arch}.attention.head_dim", "{arch}.head_dim"]`
- Fallback: Calculate as `embedding_dim // n_head` (validate divisibility)
- Used for: KV cache memory calculations

### Error Handling
- If critical metadata is missing, log warnings and attempt reasonable defaults
- If inference fails, provide clear error messages with missing keys
- Consider implementing shape-based inference as future enhancement

## KV Cache Memory Formula

### Per-Layer KV Cache Size
```
kv_bytes_per_layer = context_length × n_kv_head × head_dim × (bytes(ctk) + bytes(ctv))
```

### Data Type Size Mapping
- `fp16`: 2 bytes
- `bf16`: 2 bytes  
- `fp32`: 4 bytes
- `q8_0`: 1 byte
- `q5_0`: 5.5 bytes
- `q5_1`: 5.0 bytes
- Unknown types: Default to 2 bytes (fp16) with warning

### Total KV Cache Reservation
- Calculate per-layer size using formula above
- Multiply by number of layers that will be allocated to each GPU
- Reserve this amount before allocating any tensors
- Distribute proportionally based on expected layer allocation and `--gpu-percentage` settings

### Context Parameters
- `--context`: Context length in tokens (default: 2048)
- `-ctk`: KV cache K data type (default: f16)
- `-ctv`: KV cache V data type (default: f16)

## Block Grouping Rules

### Block Index Extraction
- Use universal regex pattern: `r'(?:^|[^0-9])(\d+)(?:\.|$)'`
- Extract the first number found in tensor name as block index
- Examples:
  - `blk.0.attn_k.weight` → block 0
  - `layers.15.attention.weight` → block 15
  - `transformer.h.42.attn.weight` → block 42

### Grouping Strategy
- Group all tensors with the same block index together
- Tensors without extractable block index go to "global" group
- Global group is allocated after all numbered blocks
- Block allocation order: 0, 1, 2, ..., global

### Co-location Enforcement
- All tensors in the same block must be allocated to the same GPU
- Never split a block across multiple GPUs
- If a block doesn't fit on current GPU, move entire block to next GPU

## GPU Capacity Sources

### Real GPU Detection (`--use-system-gpus`)
- Use NVML (NVIDIA Management Library) to enumerate NVIDIA GPUs
- Extract total VRAM capacity for each detected GPU
- Apply `--gpu-percentage` limits to determine usable VRAM
- Mutually exclusive with `--gpu-vram` option

### Hypothetical GPU Configuration (`--gpu-vram`)
- Format: `--gpu-vram 0=6,1=10` (index=GB pairs)
- Allows testing allocation strategies without requiring specific hardware
- Useful for planning deployments and dry runs
- Mutually exclusive with `--use-system-gpus` option

### GPU Percentage Limits (`--gpu-percentage`)
- Format: `--gpu-percentage 0=20,1=80,90`
- Index-specific overrides: GPU 0 uses 20%, GPU 1 uses 80%
- Default percentage: 90% (applies to GPUs without specific overrides)
- Applied to both real and hypothetical GPU configurations

### Validation Rules
- Exactly one of `--use-system-gpus` or `--gpu-vram` must be specified
- GPU percentage values must be between 1 and 100
- GPU indices in percentage overrides must correspond to available GPUs

## Allocation Algorithm

### Phase 1: KV Cache Reservation
1. Calculate total KV cache requirement using formula
2. Estimate layer distribution across GPUs (initially even split)
3. Reserve proportional KV cache space on each GPU
4. Adjust reservations based on `--gpu-percentage` limits

### Phase 2: Tensor Grouping
1. Parse all tensors from GGUF file
2. Extract block index for each tensor using regex
3. Classify each tensor by priority category (attention, ffn, gate, norm, other)
4. Group tensors by block index
5. Sort blocks numerically (0, 1, 2, ..., global)

### Phase 3: Block Allocation
1. For each block in order:
   2. Calculate total size of all tensors in block
   3. Try to allocate entire block to current GPU
   4. If block doesn't fit, move to next GPU
   5. If no GPU can fit the block, fail with error
6. Within each block, allocate tensors by priority order
7. Track remaining capacity on each GPU

### Phase 4: Spillover Handling
- When a GPU reaches capacity, move to next available GPU
- Never split a block across GPUs
- Maintain allocation order: attention → ffn → gate → norm → other
- Fail gracefully if total tensor size exceeds total available VRAM

### Allocation Constraints
- Respect GPU percentage limits at all times
- Maintain KV cache reservations (never allocate tensor space to reserved KV space)
- Preserve block co-location (entire blocks on same GPU)
- Follow priority ordering within blocks

## CLI Interface

### Primary Commands

#### `override` Command
Generate tensor override flags for llama.cpp and ik_llama.cpp.

**Required Arguments**
- `gguf`: Path or URL to GGUF model file

**GPU Configuration (Mutually Exclusive)**
- `--use-system-gpus`: Detect and use installed NVIDIA GPUs
- `--gpu-vram <index=GB,...>`: Specify hypothetical GPU configurations

**Model Parameters**
- `--context <number>`: Context size in tokens (default: 2048)
- `-ctk <type>`: KV cache K data type (default: f16)
- `-ctv <type>`: KV cache V data type (default: f16)

**GPU Allocation**
- `--gpu-percentage <index=percentage,...|percentage>`: VRAM usage limits per GPU (default: 90)

**Utility Options**
- `--help`: Show help message and exit
- `--version`: Show version information and exit
- `--verbose`: Enable verbose output for debugging

#### `check-gpus` Command
Utility command to check available GPUs on the system.

### Input Validation
- GGUF file/URL must be accessible and valid
- Context size must be positive integer
- Data types must be valid (f16, bf16, f32, q8_0, q5_1)
- GPU percentage values must be 1-100
- GPU indices must be valid for specified configuration

### Error Messages
- Clear, actionable error messages for validation failures
- Specific guidance for common issues (missing NVML, invalid GGUF, etc.)
- Verbose mode provides detailed allocation step information

## Output Format

### Runtime-Specific Formatters

#### LlamaCppOutputFormatter (Default)
Primary output formatter for llama.cpp integration:
```bash
# llama.cpp Tensor Override Flags
-ot "^blk\.0\.attn_k\.weight$=CUDA0"
-ot "^blk\.0\.attn_v\.weight$=CUDA0"
-ot "^blk\.1\.attn_k\.weight$=CUDA1"

# Allocation Summary
# GPU 0: 3.2GB tensors + 1.8GB KV cache = 83% (147 tensors)
```

#### GenericOutputFormatter (Alternative)
Generic tensor→GPU mapping for flexibility and tool chaining:
```
tensor_name_1:gpu_0 tensor_name_2:gpu_0 tensor_name_3:gpu_1 ...
```

### Allocation Summary
Both formatters provide comprehensive summary including:
- Total tensors processed and allocated
- VRAM usage per GPU (allocated + reserved for KV cache)
- KV cache reservation per GPU
- Number of blocks allocated per GPU
- Any warnings or fallbacks used during processing

### Example Output
```
# Tensor Allocations
blk.0.attn_k.weight:gpu_0 blk.0.attn_v.weight:gpu_0 blk.0.ffn_down.weight:gpu_0
blk.1.attn_k.weight:gpu_1 blk.1.attn_v.weight:gpu_1 blk.1.ffn_down.weight:gpu_1

# Allocation Summary
GPU 0: 3.2GB tensors + 1.8GB KV cache = 5.0GB / 6.0GB (83%)
GPU 1: 3.1GB tensors + 1.9GB KV cache = 5.0GB / 10.0GB (50%)
Total: 147 tensors allocated across 2 GPUs
Blocks: GPU 0 has layers 0-15, GPU 1 has layers 16-31
```

### Verbose Output
When `--verbose` is enabled, include:
- Metadata extraction details
- KV cache calculation breakdown
- Block grouping information
- Step-by-step allocation decisions
- GPU capacity tracking

## Testing Strategy

### Unit Tests (Core Logic)
- **Metadata extraction**: Test tolerant key mapping with synthetic GGUF metadata
- **Tensor classification**: Verify priority assignment for various tensor names
- **Block grouping**: Test regex pattern with diverse tensor naming conventions
- **KV cache calculation**: Validate formula with different architectures and parameters
- **Allocation algorithm**: Test edge cases (exact fits, overflows, empty blocks)

### Integration Tests (Real Files)
- **Parser integration**: Use 1-2 small public GGUF files (TinyLlama, small Qwen)
- **End-to-end validation**: Parse → classify → allocate → output generation
- **Error handling**: Test with malformed or incomplete GGUF files

### Property-Based Testing
- **Block co-location**: Verify same-block tensors always on same GPU
- **VRAM limits**: Ensure allocation never exceeds GPU capacity limits
- **Priority ordering**: Confirm higher priority tensors allocated before lower priority
- **KV cache preservation**: Verify KV space is always reserved

### Test Fixtures
- Synthetic GGUF metadata for controlled testing
- Golden output files for regression testing
- Edge case configurations (single GPU, exact capacity matches, etc.)

## Future Enhancements

### Shape-Derived Inference (Later)
- Derive head_dim, n_head, n_kv_head from Q/K/V tensor shapes when metadata missing
- Validate consistency between metadata and tensor shapes
- Provide fallback when architecture is unknown or metadata incomplete

### Block Splitting Allocation (Stretch Goal)
- Allow splitting blocks across GPUs when strict co-location prevents optimal VRAM utilization
- Implement as optional mode with clear performance trade-off warnings
- Maintain priority ordering within split blocks

### Additional Runtime-Specific Formatters (Future)
- **ik_llama.cpp formatter**: Generate appropriate flags for ik_llama.cpp syntax
- **Generic JSON**: Machine-readable allocation data for tool integration
- **Improved pattern optimization**: More efficient regex patterns for tensor groups

### Advanced Features (Future)
- Multi-node allocation for distributed inference
- Dynamic allocation based on actual memory usage patterns
- Integration with model serving frameworks
- Support for non-NVIDIA GPUs (AMD, Intel)

## Implementation Notes

### Dependencies
- `gguf-parser>=0.1.1`: GGUF file parsing
- `typer>=0.19.1`: CLI framework
- `nvidia-ml-py>=13.580.82`: NVIDIA GPU detection
- Standard library: `re`, `json`, `pathlib`, etc.

### Performance Considerations
- Lazy loading of GGUF files for large models
- Efficient tensor size calculations (avoid loading full tensor data)
- Memory-efficient allocation algorithm for models with thousands of tensors

### Error Recovery
- Graceful degradation when optional features unavailable (NVML, etc.)
- Clear error messages with suggested fixes
- Validate inputs early to fail fast

### Extensibility
- Pluggable architecture detection system
- Configurable tensor classification rules
- Modular output formatters
- Customizable allocation strategies

This specification provides a complete foundation for implementing the GGUF Tensor Overrider while maintaining flexibility for future enhancements and optimizations.