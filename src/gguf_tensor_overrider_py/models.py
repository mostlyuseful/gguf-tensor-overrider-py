"""Data models for GGUF tensor allocation system."""

from __future__ import annotations

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
        pattern = r'(?:^|[^0-9])(\d+)(?:\.|$)'
        match = re.search(pattern, self.name)
        return int(match.group(1)) if match else None
    
    def _classify_priority(self) -> TensorPriority:
        """Classify tensor by priority based on name patterns."""
        name_lower = self.name.lower()
        
        # Check for norm first (since layernorm contains both 'attention' and 'norm')
        if 'norm' in name_lower:
            return TensorPriority.NORM
        
        # Attention tensors (highest priority)
        if 'attention' in name_lower or 'attn' in name_lower:
            return TensorPriority.ATTENTION
        
        # Feed-forward tensors (exclude expert/gate/norm)
        if ('ffn' in name_lower or 'feed_forward' in name_lower):
            if not any(excl in name_lower for excl in ['exp', 'expert', 'gate', 'norm']):
                return TensorPriority.FEED_FORWARD
        
        # Gate tensors
        if 'gate' in name_lower:
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
        return self.usable_vram_bytes - self.kv_cache_reserved_bytes - self.allocated_bytes
    
    @property
    def utilization_percentage(self) -> float:
        """Current VRAM utilization as percentage of usable VRAM."""
        return (self.allocated_bytes + self.kv_cache_reserved_bytes) / self.usable_vram_bytes * 100.0
    
    def can_fit_tensor(self, size_bytes: int) -> bool:
        """Check if tensor of given size can fit in available space."""
        return self.available_bytes >= size_bytes
    
    def can_fit_block(self, block_tensors: List[TensorInfo]) -> bool:
        """Check if entire block can fit in available space."""
        total_size = sum(tensor.size_bytes for tensor in block_tensors)
        return self.available_bytes >= total_size
    
    def allocate_tensor(self, tensor: TensorInfo) -> None:
        """Allocate a tensor to this GPU."""
        if not self.can_fit_tensor(tensor.size_bytes):
            raise ValueError(f"Cannot fit tensor {tensor.name} ({tensor.size_bytes} bytes) "
                           f"in GPU {self.gpu_id} (available: {self.available_bytes} bytes)")
        
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
    
    def __post_init__(self) -> None:
        """Calculate head dimension if not provided."""
        if self.head_dim is None:
            if self.embedding_dim % self.n_heads != 0:
                raise ValueError(f"Embedding dimension {self.embedding_dim} not divisible "
                               f"by number of heads {self.n_heads}")
            self.head_dim = self.embedding_dim // self.n_heads


@dataclass
class KVCacheConfig:
    """KV cache configuration and memory requirements."""
    context_length: int
    k_dtype: DataType
    v_dtype: DataType
    
    def bytes_per_layer(self, metadata: ModelMetadata) -> int:
        """Calculate KV cache bytes per layer."""
        if metadata.head_dim is None:
            raise ValueError("Head dimension not available for KV cache calculation")
        return (self.context_length * 
                metadata.n_kv_heads * 
                metadata.head_dim * 
                (self.k_dtype.bytes_per_element + self.v_dtype.bytes_per_element))
    
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
            raise ValueError(f"Tensor {tensor.name} block ID {tensor.block_id} "
                           f"doesn't match block group ID {self.block_id}")
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
    
    def allocate_tensor_to_gpu(self, tensor: TensorInfo, gpu_id: int) -> None:
        """Allocate a tensor to specific GPU and update mappings."""
        if gpu_id not in self.gpu_allocations:
            raise ValueError(f"GPU {gpu_id} not found in available GPUs")
        
        gpu = self.gpu_allocations[gpu_id]
        gpu.allocate_tensor(tensor)
        self.tensor_gpu_mapping[tensor.name] = gpu_id
    
    def add_warning(self, message: str) -> None:
        """Add a warning message to the result."""
        self.warnings.append(message)
    
    @property
    def total_allocated_bytes(self) -> int:
        """Total bytes allocated across all GPUs."""
        return sum(gpu.allocated_bytes for gpu in self.gpu_allocations.values())
    
    @property
    def total_kv_cache_bytes(self) -> int:
        """Total KV cache bytes reserved across all GPUs."""
        return sum(gpu.kv_cache_reserved_bytes for gpu in self.gpu_allocations.values())
    
    @property
    def allocation_summary(self) -> Dict[str, Any]:
        """Generate allocation summary for output."""
        return {
            'total_tensors': len(self.tensor_gpu_mapping),
            'unallocated_tensors': len(self.unallocated_tensors),
            'total_allocated_bytes': self.total_allocated_bytes,
            'total_kv_cache_bytes': self.total_kv_cache_bytes,
            'gpu_utilization': {
                gpu_id: {
                    'allocated_bytes': gpu.allocated_bytes,
                    'kv_cache_bytes': gpu.kv_cache_reserved_bytes,
                    'utilization_percent': gpu.utilization_percentage,
                    'allocated_blocks': sorted([b for b in gpu.allocated_blocks if b is not None]),
                    'tensor_count': len(gpu.allocated_tensors)
                }
                for gpu_id, gpu in self.gpu_allocations.items()
            },
            'warnings': self.warnings
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
            usable_percentage=self.percentage
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
            layer_count_keys=[f"{arch}.block_count", f"{arch}.n_layer", f"{arch}.layer_count"],
            n_heads_keys=[f"{arch}.attention.head_count", f"{arch}.n_head"],
            n_kv_heads_keys=[f"{arch}.attention.head_count_kv", f"{arch}.n_kv_head", f"{arch}.rope.n_kv_head"],
            head_dim_keys=[f"{arch}.attention.head_dim", f"{arch}.head_dim"]
        )