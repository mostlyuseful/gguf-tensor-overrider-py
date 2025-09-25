"""Class hierarchy and architecture design for GGUF Tensor Overrider."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union
from urllib.parse import urlparse

from gguf_parser import GGUFParser
from .httpggufparser import HttpGGUFParser

from .models import (
    AllocationResult, ArchitectureKeys, BlockGroup, DataType,
    GPUConfiguration, KVCacheConfig, ModelMetadata, TensorInfo
)


# ============================================================================
# Abstract Interfaces and Protocols
# ============================================================================

class GPUDiscoveryProtocol(Protocol):
    """Protocol for GPU discovery implementations."""
    
    def discover_gpus(self) -> List[GPUConfiguration]:
        """Discover available GPUs and return their configurations."""
        ...


class OutputFormatterProtocol(Protocol):
    """Protocol for output formatting implementations."""
    
    def format_allocation(self, result: AllocationResult) -> str:
        """Format allocation result for specific runtime."""
        ...


# ============================================================================
# Core Service Classes
# ============================================================================

class MetadataExtractor:
    """Extracts and validates model metadata from GGUF files."""
    
    def extract_metadata(self, gguf_parser: GGUFParser) -> ModelMetadata:
        """Extract model metadata from parsed GGUF file."""
        metadata = gguf_parser.metadata
        if metadata is None:
            raise ValueError("No metadata found in GGUF file")
        
        # Get architecture
        architecture = self._get_architecture(metadata)
        
        # Get architecture-specific keys
        arch_keys = ArchitectureKeys.for_architecture(architecture)
        
        # Extract required values with fallbacks
        embedding_dim = self._find_metadata_value(metadata, arch_keys.embedding_keys)
        n_layers = self._find_metadata_value(metadata, arch_keys.layer_count_keys)
        n_heads = self._find_metadata_value(metadata, arch_keys.n_heads_keys)
        n_kv_heads = self._find_metadata_value(metadata, arch_keys.n_kv_heads_keys)
        head_dim = self._find_metadata_value(metadata, arch_keys.head_dim_keys, required=False)
        
        if embedding_dim is None or n_layers is None or n_heads is None or n_kv_heads is None:
            raise ValueError("Required metadata values not found")
        
        return ModelMetadata(
            architecture=architecture,
            embedding_dim=embedding_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim
        )
    
    def _get_architecture(self, metadata: Dict[str, Any]) -> str:
        """Extract and validate architecture from metadata."""
        arch = metadata.get('general.architecture', '').lower()
        # Validate against a permissive allow-list of known families to provide
        # a clearer error than missing key fallbacks when architecture is unknown.
        supported_prefixes = {
            'llama', 'mistral', 'mixtral', 'qwen', 'qwen2', 'phi', 'gemma',
            'glm', 'glm4', 'glm4moe', 'baichuan', 'falcon', 'mpt', 'olmo', 'cohere',
            'deepseek', 'deepseek2'
        }
        if not any(arch.startswith(p) for p in supported_prefixes):
            raise ValueError(f"Unsupported architecture: {arch}")
        return arch
    
    def _find_metadata_value(self, metadata: Dict[str, Any], keys: List[str], required: bool = True) -> Optional[int]:
        """Find first available value from list of possible keys."""
        for key in keys:
            if key in metadata:
                value = metadata[key]
                if isinstance(value, (int, float)):
                    return int(value)
        
        if required:
            raise ValueError(f"Required metadata not found. Tried keys: {keys}")
        return None


class TensorProcessor:
    """Processes tensors from GGUF files and groups them into blocks."""
    
    def process_tensors(self, gguf_parser: GGUFParser) -> List[BlockGroup]:
        """Process all tensors and group them by block ID."""
        tensors_info = gguf_parser.tensors_info
        if tensors_info is None:
            raise ValueError("No tensor information found in GGUF file")
        
        # Create TensorInfo objects with automatic classification
        tensors = []
        for tensor_data in tensors_info:
            size_bytes = self._calculate_tensor_size(tensor_data, tensors_info)
            tensor = TensorInfo(tensor_data['name'], size_bytes)
            tensors.append(tensor)
        
        # Group tensors by block ID
        block_groups = self._group_tensors_by_block(tensors)
        
        # Sort tensors within each block by priority
        for block in block_groups:
            block.sort_by_priority()
        
        return block_groups
    
    def _calculate_tensor_size(self, tensor_data: Dict[str, Any], all_tensors: List[Dict[str, Any]]) -> int:
        """Calculate tensor size using offset differences."""
        current_offset = tensor_data['offset']
        # Respect part boundaries when present (for split GGUFs)
        current_part = tensor_data.get('part_id')
        
        # Find next tensor with higher offset within the same part (if part_id present)
        higher_offsets = [
            t['offset'] for t in all_tensors
            if t['offset'] > current_offset and (('part_id' not in t and current_part is None) or t.get('part_id') == current_part)
        ]
        
        if higher_offsets:
            next_offset = min(higher_offsets)
            return next_offset - current_offset
        else:
            # Last tensor - estimate from dimensions and type
            dimensions = tensor_data['dimensions']
            # This is a simplified calculation - in practice, we'd need proper type size mapping
            element_count = 1
            for dim in dimensions:
                element_count *= dim
            return element_count * 2  # Assume 2 bytes per element as fallback
    
    def _group_tensors_by_block(self, tensors: List[TensorInfo]) -> List[BlockGroup]:
        """Group tensors by their block ID."""
        blocks = {}
        
        for tensor in tensors:
            block_id = tensor.block_id
            if block_id not in blocks:
                blocks[block_id] = BlockGroup(block_id)
            blocks[block_id].add_tensor(tensor)
        
        # Sort blocks: numbered blocks first (0, 1, 2, ...), then global (None)
        sorted_blocks = []
        
        # Add numbered blocks in order
        numbered_blocks = [(k, v) for k, v in blocks.items() if k is not None]
        numbered_blocks.sort(key=lambda x: x[0])
        sorted_blocks.extend([block for _, block in numbered_blocks])
        
        # Add global block if it exists
        if None in blocks:
            sorted_blocks.append(blocks[None])
        
        return sorted_blocks


class GPUManager:
    """Manages GPU discovery and configuration."""
    
    def __init__(self):
        self._nvml_available = self._check_nvml_availability()
    
    def get_gpu_configurations(self, 
                              use_system_gpus: bool = False,
                              gpu_vram_config: Optional[str] = None,
                              gpu_percentages: Optional[str] = None) -> List[GPUConfiguration]:
        """Get GPU configurations from system or user specification."""
        
        if use_system_gpus and gpu_vram_config:
            raise ValueError("Cannot use both --use-system-gpus and --gpu-vram")
        
        if not use_system_gpus and not gpu_vram_config:
            raise ValueError("Must specify either --use-system-gpus or --gpu-vram")
        
        # Parse percentage overrides
        percentage_overrides = self._parse_gpu_percentages(gpu_percentages or "90")
        
        if use_system_gpus:
            return self._discover_system_gpus(percentage_overrides)
        elif gpu_vram_config is not None:
            return self._parse_hypothetical_gpus(gpu_vram_config, percentage_overrides)
        else:
            raise ValueError("Must specify either --use-system-gpus or --gpu-vram")
    
    def _check_nvml_availability(self) -> bool:
        """Check if NVML is available for GPU discovery."""
        try:
            import pynvml
            pynvml.nvmlInit()
            return True
        except (ImportError, Exception):
            return False
    
    def _discover_system_gpus(self, percentage_overrides: Dict[Union[int, str], float]) -> List[GPUConfiguration]:
        """Discover system GPUs using NVML."""
        if not self._nvml_available:
            raise RuntimeError("NVML not available. Install pynvml or use --gpu-vram instead.")
        
        try:
            import pynvml
        except ImportError:
            raise RuntimeError("pynvml not installed. Run: pip install pynvml")
        
        gpus = []
        device_count = pynvml.nvmlDeviceGetCount()
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_gb = float(mem_info.total) / (1024**3)
            
            percentage = percentage_overrides.get(i, percentage_overrides.get('default', 90.0))
            gpus.append(GPUConfiguration(i, vram_gb, percentage))
        
        return gpus
    
    def _parse_hypothetical_gpus(self, gpu_vram_config: str, 
                                percentage_overrides: Dict[Union[int, str], float]) -> List[GPUConfiguration]:
        """Parse hypothetical GPU configuration string."""
        gpus = []
        
        for gpu_spec in gpu_vram_config.split(','):
            gpu_spec = gpu_spec.strip()
            if '=' not in gpu_spec:
                raise ValueError(f"Invalid GPU spec: {gpu_spec}. Expected format: index=GB")
            
            index_str, vram_str = gpu_spec.split('=', 1)
            try:
                gpu_id = int(index_str)
                vram_gb = float(vram_str)
            except ValueError:
                raise ValueError(f"Invalid GPU spec: {gpu_spec}. Index and VRAM must be numbers.")
            
            percentage = percentage_overrides.get(gpu_id, percentage_overrides.get('default', 90.0))
            gpus.append(GPUConfiguration(gpu_id, vram_gb, percentage))
        
        return gpus
    
    def _parse_gpu_percentages(self, percentage_config: str) -> Dict[Union[int, str], float]:
        """Parse GPU percentage configuration."""
        overrides = {}
        
        for spec in percentage_config.split(','):
            spec = spec.strip()
            
            if '=' in spec:
                # Index-specific override: "0=80"
                index_str, percent_str = spec.split('=', 1)
                try:
                    gpu_id = int(index_str)
                    percentage = float(percent_str)
                    if not (1 <= percentage <= 100):
                        raise ValueError(f"Percentage must be 1-100, got {percentage}")
                    overrides[gpu_id] = percentage
                except ValueError as e:
                    raise ValueError(f"Invalid percentage spec: {spec}. {e}")
            else:
                # Default percentage: "90"
                try:
                    percentage = float(spec)
                    if not (1 <= percentage <= 100):
                        raise ValueError(f"Percentage must be 1-100, got {percentage}")
                    overrides['default'] = percentage
                except ValueError:
                    raise ValueError(f"Invalid default percentage: {spec}")
        
        return overrides


class TensorAllocator:
    """Core tensor allocation algorithm implementation."""
    
    def allocate_tensors(self, 
                        block_groups: List[BlockGroup],
                        gpu_configs: List[GPUConfiguration],
                        kv_config: KVCacheConfig,
                        metadata: ModelMetadata) -> AllocationResult:
        """Allocate tensors across GPUs using the specified algorithm."""
        
        result = AllocationResult()
        
        # Set KV config and metadata for CPU calculations
        result.set_kv_config(kv_config, metadata)
        
        # Initialize GPU capacities
        for config in gpu_configs:
            gpu_capacity = config.to_gpu_capacity()
            result.gpu_allocations[config.gpu_id] = gpu_capacity
        
        # Reserve KV cache space
        self._reserve_kv_cache(result, kv_config, metadata)
        
        # Allocate blocks in order
        current_gpu_index = 0
        gpu_ids = sorted(result.gpu_allocations.keys())
        
        for block in block_groups:
            allocated = False
            
            # Try to allocate entire block to current GPU
            for attempt in range(len(gpu_ids)):
                gpu_id = gpu_ids[(current_gpu_index + attempt) % len(gpu_ids)]
                gpu = result.gpu_allocations[gpu_id]
                
                if gpu.can_fit_block(block.tensors):
                    # Allocate all tensors in block to this GPU
                    for tensor in block.tensors:
                        result.allocate_tensor_to_gpu(tensor, gpu_id)
                    
                    allocated = True
                    current_gpu_index = (current_gpu_index + attempt) % len(gpu_ids)
                    break
            
            if not allocated:
                # Block doesn't fit on any GPU
                result.unallocated_tensors.extend(block.tensors)
                result.add_warning(f"Block {block.block_id} ({block.total_size_bytes/1024**2:.1f} MB) "
                                 f"doesn't fit on any available GPU")
        
        return result
    
    def _reserve_kv_cache(self, result: AllocationResult, 
                         kv_config: KVCacheConfig, metadata: ModelMetadata) -> None:
        """Reserve KV cache space proportionally across GPUs."""
        total_kv_bytes = kv_config.total_bytes(metadata)
        total_usable_vram = sum(gpu.usable_vram_bytes for gpu in result.gpu_allocations.values())
        
        for gpu in result.gpu_allocations.values():
            # Proportional allocation based on usable VRAM
            proportion = gpu.usable_vram_bytes / total_usable_vram
            gpu.kv_cache_reserved_bytes = int(total_kv_bytes * proportion)


# ============================================================================
# Output Formatters
# ============================================================================

class GenericOutputFormatter:
    """Default output formatter producing generic tensor→GPU mapping."""
    
    def format_allocation(self, result: AllocationResult) -> str:
        """Format allocation result as generic mapping."""
        lines = []
        
        # Tensor mappings
        if result.tensor_gpu_mapping:
            lines.append("# Tensor Allocations")
            mapping_strs = [f"{tensor}:gpu_{gpu_id}" 
                          for tensor, gpu_id in result.tensor_gpu_mapping.items()]
            lines.append(" ".join(mapping_strs))
            lines.append("")
        
        # Summary
        summary = result.allocation_summary
        lines.append("# Allocation Summary")
        lines.append(f"Total tensors: {summary['total_tensors']}")
        lines.append(f"Unallocated: {summary['unallocated_tensors']}")
        
        for gpu_id, gpu_info in summary['gpu_utilization'].items():
            vram_gb = gpu_info['allocated_bytes'] / (1024**3)
            kv_gb = gpu_info['kv_cache_bytes'] / (1024**3)
            util_pct = gpu_info['utilization_percent']
            tensor_count = gpu_info['tensor_count']
            
            lines.append(f"GPU {gpu_id}: {vram_gb:.1f}GB tensors + {kv_gb:.1f}GB KV cache "
                        f"= {util_pct:.1f}% ({tensor_count} tensors)")
            
            if gpu_info['allocated_blocks']:
                blocks_str = ", ".join(map(str, gpu_info['allocated_blocks']))
                lines.append(f"  Blocks: {blocks_str}")

        # Total offload to CPU RAM (unallocated tensors + CPU KV cache)
        if result.unallocated_tensors or result.cpu_kv_cache_bytes > 0:
            unalloc_bytes = sum(t.size_bytes for t in result.unallocated_tensors)
            unalloc_gb = unalloc_bytes / (1024**3)
            cpu_kv_gb = result.cpu_kv_cache_bytes / (1024**3)
            total_cpu_gb = result.total_cpu_bytes / (1024**3)
            
            lines.append(f"CPU offload (unallocated tensors): {unalloc_gb:.1f}GB")
            if result.cpu_kv_cache_bytes > 0:
                lines.append(f"CPU KV cache (unallocated layers): {cpu_kv_gb:.1f}GB")
            lines.append(f"Total CPU memory required: {total_cpu_gb:.1f}GB")
        
        # Warnings
        if summary['warnings']:
            lines.append("")
            lines.append("# Warnings")
            for warning in summary['warnings']:
                lines.append(f"⚠️  {warning}")
        
        return "\n".join(lines)


class LlamaCppOutputFormatter:
    """Output formatter for llama.cpp tensor override flags."""
    
    def format_allocation(self, result: AllocationResult) -> str:
        """Format allocation result as llama.cpp -ot flags."""
        lines = []
        
        # Generate tensor override flags
        if result.tensor_gpu_mapping:
            lines.append("# llama.cpp Tensor Override Flags")
            
            # Group tensors by GPU for efficient override generation
            gpu_tensors = {}
            for tensor_name, gpu_id in result.tensor_gpu_mapping.items():
                if gpu_id not in gpu_tensors:
                    gpu_tensors[gpu_id] = []
                gpu_tensors[gpu_id].append(tensor_name)
            
            # Generate override flags for each GPU
            override_flags = []
            for gpu_id in sorted(gpu_tensors.keys()):
                device_name = f"CUDA{gpu_id}" if gpu_id >= 0 else "CPU"
                tensor_names = gpu_tensors[gpu_id]
                
                # Try to create efficient regex patterns for groups of similar tensors
                patterns = self._generate_tensor_patterns(tensor_names)
                
                for pattern in patterns:
                    # Use pattern directly (already escaped by _generate_tensor_patterns)
                    override_flags.append(f'-ot "{pattern}={device_name}"')
            
            # Add flags to output
            for flag in override_flags:
                lines.append(flag)
            
            lines.append("")
        
        # Add summary information as comments
        summary = result.allocation_summary
        lines.append("# Allocation Summary")
        lines.append(f"# Total tensors: {summary['total_tensors']}")
        lines.append(f"# Unallocated: {summary['unallocated_tensors']}")
        
        for gpu_id, gpu_info in summary['gpu_utilization'].items():
            vram_gb = gpu_info['allocated_bytes'] / (1024**3)
            kv_gb = gpu_info['kv_cache_bytes'] / (1024**3)
            util_pct = gpu_info['utilization_percent']
            tensor_count = gpu_info['tensor_count']
            
            lines.append(f"# GPU {gpu_id}: {vram_gb:.1f}GB tensors + {kv_gb:.1f}GB KV cache "
                        f"= {util_pct:.1f}% ({tensor_count} tensors)")
            
            if gpu_info['allocated_blocks']:
                blocks_str = ", ".join(map(str, gpu_info['allocated_blocks']))
                lines.append(f"#   Blocks: {blocks_str}")

        # Total offload to CPU RAM (unallocated tensors + CPU KV cache)
        if result.unallocated_tensors:
            unalloc_bytes = sum(t.size_bytes for t in result.unallocated_tensors)
        else:
            unalloc_bytes = 0
        unalloc_gb = unalloc_bytes / (1024**3)
        cpu_kv_gb = result.cpu_kv_cache_bytes / (1024**3)
        total_cpu_gb = result.total_cpu_bytes / (1024**3)
        
        lines.append(f"# CPU offload (unallocated tensors): {unalloc_gb:.1f}GB")
        if result.cpu_kv_cache_bytes > 0:
            lines.append(f"# CPU KV cache (unallocated layers): {cpu_kv_gb:.1f}GB")
        lines.append(f"# Total CPU memory required: {total_cpu_gb:.1f}GB")
        
        # Add warnings as comments
        if summary['warnings']:
            lines.append("#")
            lines.append("# Warnings:")
            for warning in summary['warnings']:
                lines.append(f"# ⚠️  {warning}")
        
        return "\n".join(lines)
    
    def _generate_tensor_patterns(self, tensor_names: List[str]) -> List[str]:
        """Generate efficient regex patterns for tensor names."""
        if not tensor_names:
            return []
        
        # For now, use individual tensor patterns
        # Future optimization: group similar patterns
        patterns = []
        
        for tensor_name in sorted(tensor_names):
            # Escape special regex characters and create exact match pattern
            escaped_name = self._escape_tensor_name(tensor_name)
            patterns.append(f"^{escaped_name}$")
        
        return patterns
    
    def _escape_tensor_name(self, tensor_name: str) -> str:
        """Escape special regex characters in tensor names."""
        # Escape special regex characters that might appear in tensor names
        special_chars = r'\.^$*+?{}[]|()'
        escaped = tensor_name
        
        for char in special_chars:
            escaped = escaped.replace(char, f"\\{char}")
        
        return escaped


# ============================================================================
# Main Application Service
# ============================================================================

@dataclass
class AllocationRequest:
    """Request parameters for tensor allocation."""
    gguf_path: Union[str, Path]
    use_system_gpus: bool = False
    gpu_vram_config: Optional[str] = None
    gpu_percentages: Optional[str] = None
    context_length: int = 2048
    k_dtype: DataType = DataType.F16
    v_dtype: DataType = DataType.F16
    verbose: bool = False


class GGUFTensorOverrider:
    """Main application service orchestrating the allocation process."""
    
    def __init__(
        self,
        metadata_extractor: Optional[MetadataExtractor] = None,
        tensor_processor: Optional[TensorProcessor] = None,
        gpu_manager: Optional[GPUManager] = None,
        allocator: Optional[TensorAllocator] = None,
        output_formatter: Optional[OutputFormatterProtocol] = None,
    ):
        self.metadata_extractor = metadata_extractor or MetadataExtractor()
        self.tensor_processor = tensor_processor or TensorProcessor()
        self.gpu_manager = gpu_manager or GPUManager()
        self.allocator = allocator or TensorAllocator()
        self.output_formatter = output_formatter or LlamaCppOutputFormatter()
        
    def process_allocation_request(self, request: AllocationRequest) -> str:
        """Process complete allocation request and return formatted output."""
        try:
            # Parse GGUF file
            if request.verbose:
                print(f"Loading GGUF file: {request.gguf_path}")
            
            gguf_parser = self._load_gguf_file(request.gguf_path)

            # If this GGUF is split into multiple parts, parse and merge tensors from all parts
            try:
                split_count = int(gguf_parser.metadata.get('split.count', 1)) if gguf_parser.metadata else 1
            except Exception:
                split_count = 1
            if split_count > 1:
                if request.verbose:
                    print(f"Detected split GGUF with split.count={split_count}. Discovering and parsing all parts...")
                part_paths = self._discover_split_parts(str(request.gguf_path), split_count)
                if request.verbose:
                    print(f"Found {len(part_paths)} parts")
                combined_tensors: List[Dict[str, Any]] = []
                seen_names: set[str] = set()
                for idx, part_path in enumerate(part_paths):
                    if request.verbose:
                        print(f"Parsing part {idx+1}/{len(part_paths)}: {part_path}")
                    part_parser = HttpGGUFParser(part_path)
                    part_parser.parse()
                    # Optional: verify consistent split.count
                    # Merge tensors, tagging with part_id for size calculation segregation
                    if part_parser.tensors_info:
                        for t in part_parser.tensors_info:
                            # Copy to avoid mutating original structures
                            td = dict(t)
                            td['part_id'] = idx
                            name = td.get('name')
                            if isinstance(name, str) and name not in seen_names:
                                combined_tensors.append(td)
                                seen_names.add(name)
                if not combined_tensors:
                    raise RuntimeError("Split GGUF detected but no tensors were parsed from parts")
                # Replace tensors_info on primary parser with the aggregated list
                gguf_parser.tensors_info = combined_tensors
            
            # Extract metadata
            if request.verbose:
                print("Extracting model metadata...")
            
            metadata = self.metadata_extractor.extract_metadata(gguf_parser)
            
            # Process tensors
            if request.verbose:
                print("Processing tensors and grouping by blocks...")
            
            block_groups = self.tensor_processor.process_tensors(gguf_parser)
            
            # Get GPU configurations
            if request.verbose:
                print("Configuring GPUs...")
            
            gpu_configs = self.gpu_manager.get_gpu_configurations(
                use_system_gpus=request.use_system_gpus,
                gpu_vram_config=request.gpu_vram_config,
                gpu_percentages=request.gpu_percentages
            )
            
            # Create KV cache config
            kv_config = KVCacheConfig(
                context_length=request.context_length,
                k_dtype=request.k_dtype,
                v_dtype=request.v_dtype
            )
            
            # Perform allocation
            if request.verbose:
                print("Allocating tensors to GPUs...")
            
            result = self.allocator.allocate_tensors(block_groups, gpu_configs, kv_config, metadata)
            
            # Format output
            return self.output_formatter.format_allocation(result)
            
        except Exception as e:
            if request.verbose:
                import traceback
                traceback.print_exc()
            raise RuntimeError(f"Allocation failed: {e}") from e
    
    def _load_gguf_file(self, gguf_path: Union[str, Path]) -> GGUFParser:
        """Load GGUF file from local path or URL."""
        path_str = str(gguf_path)
        
        # Use HttpGGUFParser which supports both URLs and local files
        parser = HttpGGUFParser(path_str)
        parser.parse()
        return parser

    def _discover_split_parts(self, gguf_path: Union[str, Path], split_count: int) -> List[str]:
        """Discover all split GGUF part file paths/URLs.
        
        Looks for pattern '-<index>-of-<total>.gguf'. Uses the width of <index> for zero-padding.
        For HTTP URLs, this pattern must be present to construct sibling URLs. For local files,
        if pattern is missing, attempts to scan the directory for matching files with same prefix.
        """
        path_str = str(gguf_path)
        is_http = urlparse(path_str).scheme in ('http', 'https')

        # Regex to find the split suffix
        m = re.search(r"-(\d+)-of-(\d+)\.gguf$", path_str)
        if m:
            idx_str, total_str = m.group(1), m.group(2)
            width = len(idx_str)
            # Use split_count from metadata when provided, otherwise fall back to parsed total
            total = split_count if split_count is not None else int(total_str)
            base = path_str[: m.start(1) - 1]  # up to the hyphen before index
            # base currently ends before '-<idx>'? Let's reconstruct carefully:
            # Split path into prefix + pattern
            prefix = path_str[: m.start()]  # before '-<idx>-of-<total>.gguf'
            suffix_after = path_str[m.end():]  # should be empty
            parts = []
            for i in range(1, int(total) + 1):
                new_idx = str(i).zfill(width)
                part = f"{prefix}-{new_idx}-of-{str(total).zfill(len(total_str))}.gguf{suffix_after}"
                parts.append(part)
            return parts

        # If no pattern in name
        if is_http:
            raise RuntimeError("split.count > 1 but URL does not contain '-<idx>-of-<total>.gguf' pattern to discover other parts")
        
        # Local file: try to scan directory
        p = Path(path_str)
        if not p.exists():
            raise RuntimeError(f"GGUF file not found: {p}")
        dir_path = p.parent
        # Attempt to derive the model prefix (filename without split suffix if it existed)
        stem = p.name
        # Collect files that match the split pattern and share the longest common prefix with this file name
        candidates = sorted([f for f in dir_path.iterdir() if f.is_file() and re.search(r"-(\d+)-of-(\d+)\.gguf$", f.name)])
        if not candidates:
            raise RuntimeError("split.count > 1 but could not find any sibling files matching '-<idx>-of-<total>.gguf'")
        # Try to filter candidates with same leading prefix segment before the split suffix
        def split_prefix(name: str) -> str:
            mm = re.search(r"-(\d+)-of-(\d+)\.gguf$", name)
            return name[: mm.start()] if mm else name
        this_prefix = split_prefix(stem)
        filtered = [str(f) for f in candidates if split_prefix(f.name) == this_prefix]
        if not filtered:
            # Fallback to all candidates; best effort
            filtered = [str(f) for f in candidates]
        # Ensure we have exactly split_count entries if possible
        if len(filtered) >= split_count:
            # Sort by index
            def idx_of(name: str) -> int:
                mm = re.search(r"-(\d+)-of-(\d+)\.gguf$", name)
                return int(mm.group(1)) if mm else 0
            filtered.sort(key=idx_of)
            return filtered[:split_count]
        return filtered