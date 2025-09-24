"""Unit tests for core class hierarchy."""

import pytest
from unittest.mock import Mock, patch

from gguf_tensor_overrider_py.core import (
    MetadataExtractor, TensorProcessor, GPUManager, TensorAllocator,
    GenericOutputFormatter, GGUFTensorOverrider, AllocationRequest
)
from gguf_tensor_overrider_py.models import (
    ModelMetadata, DataType, GPUConfiguration, KVCacheConfig,
    BlockGroup, TensorInfo, AllocationResult
)


class TestMetadataExtractor:
    """Test metadata extraction from GGUF files."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = MetadataExtractor()
    
    def test_extract_metadata_success(self):
        """Test successful metadata extraction."""
        mock_parser = Mock()
        mock_parser.metadata = {
            'general.architecture': 'llama',
            'llama.embedding_length': 4096,
            'llama.block_count': 32,
            'llama.attention.head_count': 32,
            'llama.attention.head_count_kv': 32,
            'llama.attention.head_dim': 128
        }
        
        metadata = self.extractor.extract_metadata(mock_parser)
        
        assert metadata.architecture == 'llama'
        assert metadata.embedding_dim == 4096
        assert metadata.n_layers == 32
        assert metadata.n_heads == 32
        assert metadata.n_kv_heads == 32
        assert metadata.head_dim == 128
    
    def test_extract_metadata_with_fallbacks(self):
        """Test metadata extraction using fallback keys."""
        mock_parser = Mock()
        mock_parser.metadata = {
            'general.architecture': 'qwen2',
            'qwen2.hidden_size': 2048,  # Fallback for embedding_length
            'qwen2.n_layer': 16,        # Fallback for block_count
            'qwen2.n_head': 16,         # Fallback for attention.head_count
            'qwen2.n_kv_head': 16       # Fallback for attention.head_count_kv
        }
        
        metadata = self.extractor.extract_metadata(mock_parser)
        
        assert metadata.architecture == 'qwen2'
        assert metadata.embedding_dim == 2048
        assert metadata.n_layers == 16
        assert metadata.n_heads == 16
        assert metadata.n_kv_heads == 16
        assert metadata.head_dim == 128  # Calculated: 2048 / 16
    
    def test_extract_metadata_no_metadata(self):
        """Test error when no metadata is available."""
        mock_parser = Mock()
        mock_parser.metadata = None
        
        with pytest.raises(ValueError, match="No metadata found"):
            self.extractor.extract_metadata(mock_parser)
    
    def test_extract_metadata_unsupported_architecture(self):
        """Test error with unsupported architecture."""
        mock_parser = Mock()
        mock_parser.metadata = {'general.architecture': 'unsupported_arch'}
        
        with pytest.raises(ValueError, match="Unsupported architecture"):
            self.extractor.extract_metadata(mock_parser)
    
    def test_extract_metadata_missing_required_keys(self):
        """Test error when required metadata keys are missing."""
        mock_parser = Mock()
        mock_parser.metadata = {
            'general.architecture': 'llama',
            # Missing required keys
        }
        
        with pytest.raises(ValueError, match="Required metadata not found"):
            self.extractor.extract_metadata(mock_parser)


class TestTensorProcessor:
    """Test tensor processing and block grouping."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = TensorProcessor()
    
    def test_process_tensors_success(self):
        """Test successful tensor processing."""
        mock_parser = Mock()
        mock_parser.tensors_info = [
            {'name': 'blk.0.attn_k.weight', 'offset': 0, 'dimensions': [32, 128]},
            {'name': 'blk.0.ffn_down.weight', 'offset': 8192, 'dimensions': [128, 64]},
            {'name': 'blk.1.attn_v.weight', 'offset': 16384, 'dimensions': [32, 128]},
            {'name': 'output.weight', 'offset': 24576, 'dimensions': [32, 100]},
        ]
        
        block_groups = self.processor.process_tensors(mock_parser)
        
        assert len(block_groups) == 3  # block 0, block 1, global
        
        # Check block 0
        block_0 = next(b for b in block_groups if b.block_id == 0)
        assert len(block_0.tensors) == 2
        tensor_names = [t.name for t in block_0.tensors]
        assert 'blk.0.attn_k.weight' in tensor_names
        assert 'blk.0.ffn_down.weight' in tensor_names
        
        # Check block 1
        block_1 = next(b for b in block_groups if b.block_id == 1)
        assert len(block_1.tensors) == 1
        assert block_1.tensors[0].name == 'blk.1.attn_v.weight'
        
        # Check global block
        global_block = next(b for b in block_groups if b.block_id is None)
        assert len(global_block.tensors) == 1
        assert global_block.tensors[0].name == 'output.weight'
    
    def test_process_tensors_no_tensors(self):
        """Test error when no tensor info is available."""
        mock_parser = Mock()
        mock_parser.tensors_info = None
        
        with pytest.raises(ValueError, match="No tensor information found"):
            self.processor.process_tensors(mock_parser)
    
    def test_tensor_size_calculation(self):
        """Test tensor size calculation using offset differences."""
        tensor_data = {'offset': 1000}
        all_tensors = [
            {'offset': 0},
            {'offset': 1000},  # This tensor
            {'offset': 2000},  # Next tensor
            {'offset': 3000}
        ]
        
        size = self.processor._calculate_tensor_size(tensor_data, all_tensors)
        assert size == 1000  # 2000 - 1000
    
    def test_tensor_size_calculation_last_tensor(self):
        """Test tensor size calculation for last tensor."""
        tensor_data = {'offset': 3000, 'dimensions': [100, 50]}
        all_tensors = [
            {'offset': 0},
            {'offset': 1000},
            {'offset': 2000},
            {'offset': 3000}  # This is the last tensor
        ]
        
        size = self.processor._calculate_tensor_size(tensor_data, all_tensors)
        # Should estimate from dimensions: 100 * 50 * 2 = 10000
        assert size == 10000


class TestGPUManager:
    """Test GPU discovery and configuration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = GPUManager()
    
    def test_get_gpu_configurations_hypothetical(self):
        """Test hypothetical GPU configuration parsing."""
        configs = self.manager.get_gpu_configurations(
            use_system_gpus=False,
            gpu_vram_config="0=6,1=10",
            gpu_percentages="0=80,1=90,95"
        )
        
        assert len(configs) == 2
        
        gpu0 = next(c for c in configs if c.gpu_id == 0)
        assert gpu0.total_vram_gb == 6.0
        assert gpu0.percentage == 80.0
        
        gpu1 = next(c for c in configs if c.gpu_id == 1)
        assert gpu1.total_vram_gb == 10.0
        assert gpu1.percentage == 90.0
    
    def test_get_gpu_configurations_validation_errors(self):
        """Test validation errors in GPU configuration."""
        # Both options specified
        with pytest.raises(ValueError, match="Cannot use both"):
            self.manager.get_gpu_configurations(
                use_system_gpus=True,
                gpu_vram_config="0=6"
            )
        
        # Neither option specified
        with pytest.raises(ValueError, match="Must specify either"):
            self.manager.get_gpu_configurations(
                use_system_gpus=False,
                gpu_vram_config=None
            )
    
    def test_parse_gpu_percentages(self):
        """Test GPU percentage parsing."""
        # Test mixed format
        result = self.manager._parse_gpu_percentages("0=80,1=75,90")
        
        expected = {0: 80.0, 1: 75.0, 'default': 90.0}
        assert result == expected
    
    def test_parse_gpu_percentages_validation(self):
        """Test GPU percentage validation."""
        # Invalid percentage range
        with pytest.raises(ValueError, match="Percentage must be 1-100"):
            self.manager._parse_gpu_percentages("0=150")
        
        # Invalid format
        with pytest.raises(ValueError, match="Invalid default percentage"):
            self.manager._parse_gpu_percentages("invalid")
    
    def test_discover_system_gpus_success(self):
        """Test successful system GPU discovery."""
        with patch('builtins.__import__') as mock_import:
            # Mock the pynvml import
            mock_pynvml = Mock()
            mock_pynvml.nvmlDeviceGetCount.return_value = 2
            
            mock_handle_0 = Mock()
            mock_handle_1 = Mock()
            mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = [mock_handle_0, mock_handle_1]
            
            mock_mem_info_0 = Mock()
            mock_mem_info_0.total = 8 * 1024**3  # 8GB
            mock_mem_info_1 = Mock()
            mock_mem_info_1.total = 12 * 1024**3  # 12GB
            
            mock_pynvml.nvmlDeviceGetMemoryInfo.side_effect = [mock_mem_info_0, mock_mem_info_1]
            
            def import_side_effect(name, *args, **kwargs):
                if name == 'pynvml':
                    return mock_pynvml
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = import_side_effect
            
            # Test with mocked manager that thinks NVML is available
            manager = GPUManager()
            manager._nvml_available = True
            
            configs = manager._discover_system_gpus({0: 75.0, 'default': 85.0})
            
            assert len(configs) == 2
            assert configs[0].gpu_id == 0
            assert configs[0].total_vram_gb == 8.0
            assert configs[0].percentage == 75.0
            assert configs[1].gpu_id == 1
            assert configs[1].total_vram_gb == 12.0
            assert configs[1].percentage == 85.0


class TestTensorAllocator:
    """Test tensor allocation algorithm."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.allocator = TensorAllocator()
    
    def test_allocate_tensors_simple(self):
        """Test simple tensor allocation."""
        # Create test data
        block_groups = [
            BlockGroup(0),
            BlockGroup(1)
        ]
        
        # Create tensors with correct block IDs first
        tensor0 = TensorInfo("blk.0.attn.weight", 1024**3)  # 1GB, will have block_id=0
        tensor1 = TensorInfo("blk.1.attn.weight", 1024**3)  # 1GB, will have block_id=1
        
        # Add tensors to matching blocks
        block_groups[0].add_tensor(tensor0)
        block_groups[1].add_tensor(tensor1)
        
        gpu_configs = [
            GPUConfiguration(0, 4.0, 100.0),  # 4GB GPU
            GPUConfiguration(1, 4.0, 100.0)   # 4GB GPU
        ]
        
        kv_config = KVCacheConfig(1024, DataType.F16, DataType.F16)
        metadata = ModelMetadata("llama", 2048, 2, 16, 16, 128)
        
        result = self.allocator.allocate_tensors(block_groups, gpu_configs, kv_config, metadata)
        
        # Check that tensors were allocated
        assert len(result.tensor_gpu_mapping) == 2
        assert len(result.unallocated_tensors) == 0
        
        # Check that at least one GPU has allocations
        total_allocated = sum(gpu.allocated_bytes for gpu in result.gpu_allocations.values())
        assert total_allocated > 0
        
        # Check KV cache reservation
        for gpu in result.gpu_allocations.values():
            assert gpu.kv_cache_reserved_bytes > 0
    
    def test_allocate_tensors_overflow(self):
        """Test allocation when tensors don't fit."""
        # Create large tensor that won't fit (use name without block pattern)
        block_groups = [BlockGroup(None)]  # Global block
        huge_tensor = TensorInfo("huge_global.weight", 10 * 1024**3)  # 10GB, no block pattern
        block_groups[0].add_tensor(huge_tensor)
        
        gpu_configs = [GPUConfiguration(0, 4.0, 100.0)]  # Only 4GB available
        
        kv_config = KVCacheConfig(1024, DataType.F16, DataType.F16)
        metadata = ModelMetadata("llama", 2048, 1, 16, 16, 128)
        
        result = self.allocator.allocate_tensors(block_groups, gpu_configs, kv_config, metadata)
        
        # Should have unallocated tensors
        assert len(result.unallocated_tensors) == 1
        assert len(result.warnings) > 0


class TestGenericOutputFormatter:
    """Test output formatting."""
    
    def test_format_allocation(self):
        """Test allocation result formatting."""
        formatter = GenericOutputFormatter()
        
        # Create test allocation result
        result = AllocationResult()
        result.tensor_gpu_mapping = {
            "tensor1": 0,
            "tensor2": 0,
            "tensor3": 1
        }
        
        # Add some GPU allocations for summary
        from gguf_tensor_overrider_py.models import GPUCapacity
        gpu0 = GPUCapacity(0, 4 * 1024**3, 80.0)
        gpu0.allocated_bytes = 1 * 1024**3
        gpu0.kv_cache_reserved_bytes = 512 * 1024**2
        
        result.gpu_allocations[0] = gpu0
        result.add_warning("Test warning")
        
        output = formatter.format_allocation(result)
        
        assert "# Tensor Allocations" in output
        assert "tensor1:gpu_0" in output
        assert "# Allocation Summary" in output
        assert "Total tensors: 3" in output
        assert "# Warnings" in output
        assert "Test warning" in output


class TestGGUFTensorOverrider:
    """Test main application service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.overrider = GGUFTensorOverrider()
    
    @patch('gguf_tensor_overrider_py.core.HttpGGUFParser')
    def test_process_allocation_request_success(self, mock_http_gguf_parser):
        """Test successful allocation request processing."""
        # Mock HttpGGUFParser
        mock_parser_instance = Mock()
        mock_parser_instance.metadata = {
            'general.architecture': 'llama',
            'llama.embedding_length': 4096,
            'llama.block_count': 32,
            'llama.attention.head_count': 32,
            'llama.attention.head_count_kv': 32
        }
        mock_parser_instance.tensors_info = [
            {'name': 'test.weight', 'offset': 0, 'dimensions': [100, 100]}
        ]
        mock_http_gguf_parser.return_value = mock_parser_instance
        
        # Create test request
        request = AllocationRequest(
            gguf_path="test.gguf",
            use_system_gpus=False,
            gpu_vram_config="0=8",
            context_length=2048,
            k_dtype=DataType.F16,
            v_dtype=DataType.F16
        )
        
        output = self.overrider.process_allocation_request(request)
        
        assert "# Allocation Summary" in output
        assert isinstance(output, str)
    
    def test_load_gguf_file_not_found(self):
        """Test error when GGUF file doesn't exist."""
        from gguf_parser import GGUFParseError
        with pytest.raises(GGUFParseError, match="GGUF file not found"):
            self.overrider._load_gguf_file("nonexistent.gguf")
    
    def test_load_gguf_file_url_failure(self):
        """Test URL loading failure when URL is not accessible."""
        from gguf_tensor_overrider_py.httpfile import FileLengthError
        with pytest.raises(FileLengthError, match="Failed to get file length"):
            self.overrider._load_gguf_file("https://example.com/model.gguf")