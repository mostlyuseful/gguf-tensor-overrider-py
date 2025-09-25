"""Unit tests for data models."""

import pytest
from gguf_tensor_overrider_py.models import (
    TensorInfo, GPUCapacity, ModelMetadata, KVCacheConfig, 
    BlockGroup, AllocationResult, DataType, GPUConfiguration,
    TensorPriority, ArchitectureKeys
)


class TestDataType:
    """Test DataType enum and byte calculations."""
    
    def test_bytes_per_element(self):
        """Test byte size calculations for different data types."""
        assert DataType.F16.bytes_per_element == 2
        assert DataType.BF16.bytes_per_element == 2
        assert DataType.F32.bytes_per_element == 4
        assert DataType.Q8_0.bytes_per_element == 1
        assert DataType.Q5_0.bytes_per_element == 5.5
        assert DataType.Q5_1.bytes_per_element == 5.0

    
    def test_string_values(self):
        """Test string representation of data types."""
        assert DataType.F16.value == "f16"
        assert DataType.Q8_0.value == "q8_0"


class TestTensorInfo:
    """Test TensorInfo classification and block extraction."""
    
    def test_attention_classification(self):
        """Test attention tensor classification."""
        tensor1 = TensorInfo("blk.0.attn_k.weight", 1024)
        tensor2 = TensorInfo("layers.15.attention.q_proj.weight", 2048)
        
        assert tensor1.priority == TensorPriority.ATTENTION
        assert tensor2.priority == TensorPriority.ATTENTION
    
    def test_ffn_classification(self):
        """Test feed-forward tensor classification."""
        tensor1 = TensorInfo("blk.0.ffn_down.weight", 1024)
        tensor2 = TensorInfo("layers.15.feed_forward.intermediate.weight", 2048)
        
        assert tensor1.priority == TensorPriority.FEED_FORWARD
        assert tensor2.priority == TensorPriority.FEED_FORWARD
    
    def test_ffn_exclusions(self):
        """Test that expert/gate/norm tensors are excluded from FFN category."""
        tensor1 = TensorInfo("blk.0.ffn_gate.weight", 1024)
        tensor2 = TensorInfo("blk.0.ffn_expert.weight", 1024)
        tensor3 = TensorInfo("blk.0.ffn_norm.weight", 1024)
        
        assert tensor1.priority == TensorPriority.GATE
        assert tensor2.priority == TensorPriority.OTHER
        assert tensor3.priority == TensorPriority.NORM
    
    def test_gate_classification(self):
        """Test gate tensor classification."""
        tensor = TensorInfo("blk.0.gate_proj.weight", 1024)
        assert tensor.priority == TensorPriority.GATE
    
    def test_norm_classification(self):
        """Test norm tensor classification."""
        tensor1 = TensorInfo("blk.0.input_layernorm.weight", 1024)
        tensor2 = TensorInfo("layers.15.post_attention_layernorm.weight", 1024)
        
        assert tensor1.priority == TensorPriority.NORM
        assert tensor2.priority == TensorPriority.NORM
    
    def test_other_classification(self):
        """Test other tensor classification."""
        tensor1 = TensorInfo("token_embd.weight", 1024)
        tensor2 = TensorInfo("output.weight", 1024)
        
        assert tensor1.priority == TensorPriority.OTHER
        assert tensor2.priority == TensorPriority.OTHER
    
    def test_block_id_extraction(self):
        """Test block ID extraction from tensor names."""
        test_cases = [
            ("blk.0.attn_k.weight", 0),
            ("layers.15.attention.weight", 15),
            ("transformer.h.42.attn.weight", 42),
            ("model.layers.31.mlp.weight", 31),
            ("token_embd.weight", None),
            ("output.weight", None),
        ]
        
        for name, expected_block in test_cases:
            tensor = TensorInfo(name, 1024)
            assert tensor.block_id == expected_block, f"Failed for {name}"
    
    def test_edge_case_block_extraction(self):
        """Test edge cases for block ID extraction."""
        # Multiple numbers - should get first one
        tensor1 = TensorInfo("layer.10.head.2.weight", 1024)
        assert tensor1.block_id == 10
        
        # Number at start
        tensor2 = TensorInfo("0.embedding.weight", 1024)
        assert tensor2.block_id == 0
        
        # Number at end
        tensor3 = TensorInfo("embedding.42", 1024)
        assert tensor3.block_id == 42


class TestGPUCapacity:
    """Test GPU capacity management and allocation tracking."""
    
    def test_initialization(self):
        """Test GPU capacity initialization."""
        gpu = GPUCapacity(gpu_id=0, total_vram_bytes=8 * 1024**3, usable_percentage=80.0)
        
        assert gpu.gpu_id == 0
        assert gpu.total_vram_bytes == 8 * 1024**3
        assert gpu.usable_percentage == 80.0
        assert gpu.allocated_bytes == 0
        assert gpu.kv_cache_reserved_bytes == 0
    
    def test_usable_vram_calculation(self):
        """Test usable VRAM calculation with percentage limits."""
        gpu = GPUCapacity(gpu_id=0, total_vram_bytes=10 * 1024**3, usable_percentage=90.0)
        expected_usable = int(10 * 1024**3 * 0.90)
        
        assert gpu.usable_vram_bytes == expected_usable
    
    def test_available_bytes_calculation(self):
        """Test available bytes after KV cache reservation."""
        gpu = GPUCapacity(gpu_id=0, total_vram_bytes=8 * 1024**3, usable_percentage=100.0)
        gpu.kv_cache_reserved_bytes = 1 * 1024**3
        gpu.allocated_bytes = 2 * 1024**3
        
        expected_available = 8 * 1024**3 - 1 * 1024**3 - 2 * 1024**3
        assert gpu.available_bytes == expected_available
    
    def test_utilization_percentage(self):
        """Test utilization percentage calculation."""
        gpu = GPUCapacity(gpu_id=0, total_vram_bytes=10 * 1024**3, usable_percentage=80.0)
        gpu.kv_cache_reserved_bytes = 2 * 1024**3
        gpu.allocated_bytes = 2 * 1024**3
        
        # 4GB used out of 8GB usable = 50%
        assert gpu.utilization_percentage == 50.0
    
    def test_can_fit_tensor(self):
        """Test tensor fitting calculations."""
        gpu = GPUCapacity(gpu_id=0, total_vram_bytes=4 * 1024**3, usable_percentage=100.0)
        gpu.kv_cache_reserved_bytes = 1 * 1024**3
        
        # 3GB available
        assert gpu.can_fit_tensor(2 * 1024**3) is True
        assert gpu.can_fit_tensor(4 * 1024**3) is False
    
    def test_can_fit_block(self):
        """Test block fitting calculations."""
        gpu = GPUCapacity(gpu_id=0, total_vram_bytes=4 * 1024**3, usable_percentage=100.0)
        
        tensors = [
            TensorInfo("tensor1", 1024**3),  # 1GB
            TensorInfo("tensor2", 1024**3),  # 1GB
        ]
        
        assert gpu.can_fit_block(tensors) is True
        
        # Add more tensors to exceed capacity
        tensors.append(TensorInfo("tensor3", 3 * 1024**3))
        assert gpu.can_fit_block(tensors) is False
    
    def test_allocate_tensor_success(self):
        """Test successful tensor allocation."""
        gpu = GPUCapacity(gpu_id=0, total_vram_bytes=4 * 1024**3, usable_percentage=100.0)
        tensor = TensorInfo("test.weight", 1024**3)
        
        gpu.allocate_tensor(tensor)
        
        assert gpu.allocated_bytes == 1024**3
        assert "test.weight" in gpu.allocated_tensors
        assert tensor.block_id in gpu.allocated_blocks
    
    def test_allocate_tensor_failure(self):
        """Test tensor allocation failure when insufficient space."""
        gpu = GPUCapacity(gpu_id=0, total_vram_bytes=1 * 1024**3, usable_percentage=100.0)
        tensor = TensorInfo("huge.weight", 2 * 1024**3)
        
        with pytest.raises(ValueError, match="Cannot fit tensor"):
            gpu.allocate_tensor(tensor)


class TestModelMetadata:
    """Test model metadata and head dimension calculations."""
    
    def test_initialization_with_head_dim(self):
        """Test initialization when head_dim is provided."""
        metadata = ModelMetadata(
            architecture="llama",
            embedding_dim=4096,
            n_layers=32,
            n_heads=32,
            n_kv_heads=32,
            head_dim=128
        )
        
        assert metadata.head_dim == 128
    
    def test_head_dim_calculation(self):
        """Test automatic head dimension calculation."""
        metadata = ModelMetadata(
            architecture="llama",
            embedding_dim=4096,
            n_layers=32,
            n_heads=32,
            n_kv_heads=32
        )
        
        assert metadata.head_dim == 128  # 4096 / 32
    
    def test_head_dim_validation_error(self):
        """Test error when embedding_dim not divisible by n_heads."""
        with pytest.raises(ValueError, match="not divisible"):
            ModelMetadata(
                architecture="llama",
                embedding_dim=4097,  # Not divisible by 32
                n_layers=32,
                n_heads=32,
                n_kv_heads=32
            )


class TestKVCacheConfig:
    """Test KV cache configuration and memory calculations."""
    
    def test_bytes_per_layer(self):
        """Test KV cache bytes per layer calculation."""
        config = KVCacheConfig(
            context_length=2048,
            k_dtype=DataType.F16,
            v_dtype=DataType.F16
        )
        
        metadata = ModelMetadata(
            architecture="llama",
            embedding_dim=4096,
            n_layers=32,
            n_heads=32,
            n_kv_heads=32,
            head_dim=128
        )
        
        expected = 2048 * 32 * 128 * (2 + 2)  # context * kv_heads * head_dim * (k_bytes + v_bytes)
        assert config.bytes_per_layer(metadata) == expected
    
    def test_total_bytes(self):
        """Test total KV cache calculation."""
        config = KVCacheConfig(
            context_length=1024,
            k_dtype=DataType.F16,
            v_dtype=DataType.F16
        )
        
        metadata = ModelMetadata(
            architecture="llama",
            embedding_dim=2048,
            n_layers=16,
            n_heads=16,
            n_kv_heads=16,
            head_dim=128
        )
        
        per_layer = config.bytes_per_layer(metadata)
        expected_total = per_layer * 16
        assert config.total_bytes(metadata) == expected_total
    
    def test_different_k_v_dtypes(self):
        """Test KV cache with different K and V data types."""
        config = KVCacheConfig(
            context_length=1024,
            k_dtype=DataType.Q8_0,  # 1 byte
            v_dtype=DataType.F32    # 4 bytes
        )
        
        metadata = ModelMetadata(
            architecture="llama",
            embedding_dim=1024,
            n_layers=8,
            n_heads=8,
            n_kv_heads=8,
            head_dim=128
        )
        
        expected = 1024 * 8 * 128 * (1 + 4)  # Mixed dtype sizes
        assert config.bytes_per_layer(metadata) == expected
    
    def test_head_dim_none_error(self):
        """Test error when head_dim is None."""
        config = KVCacheConfig(
            context_length=1024,
            k_dtype=DataType.F16,
            v_dtype=DataType.F16
        )
        
        # Create metadata with head_dim explicitly set to None
        metadata = ModelMetadata(
            architecture="llama",
            embedding_dim=4096,
            n_layers=32,
            n_heads=32,
            n_kv_heads=32,
            head_dim=128
        )
        metadata.head_dim = None  # Force None after initialization
        
        with pytest.raises(ValueError, match="Head dimension not available"):
            config.bytes_per_layer(metadata)


class TestBlockGroup:
    """Test block grouping functionality."""
    
    def test_initialization(self):
        """Test block group initialization."""
        block = BlockGroup(block_id=5)
        
        assert block.block_id == 5
        assert len(block.tensors) == 0
        assert block.total_size_bytes == 0
        assert not block.is_global
    
    def test_global_block(self):
        """Test global block (no specific layer ID)."""
        block = BlockGroup(block_id=None)
        assert block.is_global
    
    def test_add_tensor_success(self):
        """Test successful tensor addition to block."""
        block = BlockGroup(block_id=10)
        tensor = TensorInfo("blk.10.attn.weight", 1024)
        
        block.add_tensor(tensor)
        
        assert len(block.tensors) == 1
        assert block.total_size_bytes == 1024
    
    def test_add_tensor_wrong_block_error(self):
        """Test error when adding tensor with wrong block ID."""
        block = BlockGroup(block_id=5)
        tensor = TensorInfo("blk.10.attn.weight", 1024)  # Wrong block ID
        
        with pytest.raises(ValueError, match="doesn't match block group ID"):
            block.add_tensor(tensor)
    
    def test_sort_by_priority(self):
        """Test sorting tensors by priority within block."""
        block = BlockGroup(block_id=0)
        
        # Add tensors in random priority order
        tensors = [
            TensorInfo("blk.0.output.weight", 512),      # OTHER (5)
            TensorInfo("blk.0.attn.weight", 1024),       # ATTENTION (1)
            TensorInfo("blk.0.norm.weight", 256),        # NORM (4)
            TensorInfo("blk.0.ffn.weight", 2048),        # FEED_FORWARD (2)
            TensorInfo("blk.0.gate.weight", 128),        # GATE (3)
        ]
        
        for tensor in tensors:
            block.add_tensor(tensor)
        
        block.sort_by_priority()
        
        # Check order: ATTENTION, FEED_FORWARD, GATE, NORM, OTHER
        priorities = [t.priority for t in block.tensors]
        expected = [
            TensorPriority.ATTENTION,
            TensorPriority.FEED_FORWARD,
            TensorPriority.GATE,
            TensorPriority.NORM,
            TensorPriority.OTHER
        ]
        assert priorities == expected


class TestAllocationResult:
    """Test allocation result tracking and summary generation."""
    
    def test_initialization(self):
        """Test allocation result initialization."""
        result = AllocationResult()
        
        assert len(result.gpu_allocations) == 0
        assert len(result.tensor_gpu_mapping) == 0
        assert len(result.unallocated_tensors) == 0
        assert len(result.warnings) == 0
    
    def test_allocate_tensor_to_gpu(self):
        """Test tensor allocation to specific GPU."""
        result = AllocationResult()
        gpu = GPUCapacity(gpu_id=0, total_vram_bytes=4 * 1024**3, usable_percentage=100.0)
        result.gpu_allocations[0] = gpu
        
        tensor = TensorInfo("test.weight", 1024**3)
        result.allocate_tensor_to_gpu(tensor, 0)
        
        assert result.tensor_gpu_mapping["test.weight"] == 0
        assert gpu.allocated_bytes == 1024**3
    
    def test_allocate_to_nonexistent_gpu_error(self):
        """Test error when allocating to non-existent GPU."""
        result = AllocationResult()
        tensor = TensorInfo("test.weight", 1024)
        
        with pytest.raises(ValueError, match="GPU 0 not found"):
            result.allocate_tensor_to_gpu(tensor, 0)
    
    def test_add_warning(self):
        """Test warning addition."""
        result = AllocationResult()
        result.add_warning("Test warning")
        
        assert len(result.warnings) == 1
        assert result.warnings[0] == "Test warning"
    
    def test_total_calculations(self):
        """Test total allocation and KV cache calculations."""
        result = AllocationResult()
        
        gpu0 = GPUCapacity(gpu_id=0, total_vram_bytes=4 * 1024**3, usable_percentage=100.0)
        gpu0.allocated_bytes = 1 * 1024**3
        gpu0.kv_cache_reserved_bytes = 512 * 1024**2
        
        gpu1 = GPUCapacity(gpu_id=1, total_vram_bytes=8 * 1024**3, usable_percentage=100.0)
        gpu1.allocated_bytes = 2 * 1024**3
        gpu1.kv_cache_reserved_bytes = 1 * 1024**3
        
        result.gpu_allocations[0] = gpu0
        result.gpu_allocations[1] = gpu1
        
        assert result.total_allocated_bytes == 3 * 1024**3
        assert result.total_kv_cache_bytes == int(1.5 * 1024**3)
    
    def test_allocation_summary(self):
        """Test allocation summary generation."""
        result = AllocationResult()
        
        gpu = GPUCapacity(gpu_id=0, total_vram_bytes=4 * 1024**3, usable_percentage=80.0)
        gpu.allocated_bytes = 1 * 1024**3
        gpu.kv_cache_reserved_bytes = 1 * 1024**3
        gpu.allocated_tensors = ["tensor1", "tensor2"]
        gpu.allocated_blocks = {0, 1, None}
        
        result.gpu_allocations[0] = gpu
        result.tensor_gpu_mapping = {"tensor1": 0, "tensor2": 0}
        result.warnings = ["Test warning"]
        
        summary = result.allocation_summary
        
        assert summary['total_tensors'] == 2
        assert summary['unallocated_tensors'] == 0
        assert summary['total_allocated_bytes'] == 1 * 1024**3
        assert summary['total_kv_cache_bytes'] == 1 * 1024**3
        assert len(summary['warnings']) == 1
        
        gpu_info = summary['gpu_utilization'][0]
        assert gpu_info['allocated_bytes'] == 1 * 1024**3
        assert gpu_info['kv_cache_bytes'] == 1 * 1024**3
        assert gpu_info['utilization_percent'] == pytest.approx(62.5)  # 2GB used / 3.2GB usable
        # New fields
        assert gpu_info['total_vram_bytes'] == 4 * 1024**3
        assert gpu_info['usable_vram_bytes'] == int(4 * 1024**3 * 0.8)
        assert gpu_info['usable_percent_limit'] == 80.0
        assert gpu_info['utilization_percent_of_total'] == pytest.approx(50.0)  # 2GB used / 4GB total
        assert gpu_info['allocated_blocks'] == [0, 1]  # None filtered out
        assert gpu_info['tensor_count'] == 2

    def test_utilization_respects_total_limit(self):
        """Ensure utilization relative to total VRAM does not exceed configured limit."""
        result = AllocationResult()
        # 24 GB total, 90% usable
        gpu = GPUCapacity(gpu_id=3, total_vram_bytes=24 * 1024**3, usable_percentage=90.0)
        # Simulate 14.8 GB tensors + 5.08 GB KV = 19.88 GB used
        gpu.allocated_bytes = int(14.8 * 1024**3)
        gpu.kv_cache_reserved_bytes = int(5.08 * 1024**3)
        result.gpu_allocations[3] = gpu
        summary = result.allocation_summary
        gpu_info = summary['gpu_utilization'][3]
        # Percent of total should be below or equal to the configured usable limit
        assert gpu_info['utilization_percent_of_total'] <= gpu_info['usable_percent_limit']


class TestGPUConfiguration:
    """Test GPU configuration and conversion to capacity."""
    
    def test_initialization(self):
        """Test GPU configuration initialization."""
        config = GPUConfiguration(gpu_id=1, total_vram_gb=8.0, percentage=85.0)
        
        assert config.gpu_id == 1
        assert config.total_vram_gb == 8.0
        assert config.percentage == 85.0
    
    def test_default_percentage(self):
        """Test default percentage value."""
        config = GPUConfiguration(gpu_id=0, total_vram_gb=6.0)
        assert config.percentage == 90.0
    
    def test_to_gpu_capacity(self):
        """Test conversion to GPU capacity."""
        config = GPUConfiguration(gpu_id=2, total_vram_gb=10.0, percentage=75.0)
        capacity = config.to_gpu_capacity()
        
        assert capacity.gpu_id == 2
        assert capacity.total_vram_bytes == 10 * 1024**3
        assert capacity.usable_percentage == 75.0
        assert capacity.allocated_bytes == 0


class TestArchitectureKeys:
    """Test architecture key mappings."""
    
    def test_for_architecture(self):
        """Test key generation for specific architecture."""
        keys = ArchitectureKeys.for_architecture("llama")
        
        expected_embedding = ["llama.embedding_length", "llama.hidden_size"]
        expected_layers = ["llama.block_count", "llama.n_layer", "llama.layer_count"]
        expected_heads = ["llama.attention.head_count", "llama.n_head"]
        expected_kv_heads = ["llama.attention.head_count_kv", "llama.n_kv_head", "llama.rope.n_kv_head"]
        expected_head_dim = [
            "llama.attention.head_dim",
            "llama.head_dim",
            "llama.attention.value_length",
            "llama.attention.key_length",
            "llama.attention.value_length_mla",
            "llama.attention.key_length_mla",
        ]
        
        assert keys.embedding_keys == expected_embedding
        assert keys.layer_count_keys == expected_layers
        assert keys.n_heads_keys == expected_heads
        assert keys.n_kv_heads_keys == expected_kv_heads
        assert keys.head_dim_keys == expected_head_dim
    
    def test_different_architectures(self):
        """Test key generation for different architectures."""
        qwen_keys = ArchitectureKeys.for_architecture("qwen2")
        
        assert "qwen2.embedding_length" in qwen_keys.embedding_keys
        assert "qwen2.block_count" in qwen_keys.layer_count_keys