"""Tests for llama.cpp output formatter."""
from gguf_tensor_overrider_py.core import LlamaCppOutputFormatter
from gguf_tensor_overrider_py.models import AllocationResult, GPUCapacity


class TestLlamaCppOutputFormatter:
    """Test llama.cpp output formatter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = LlamaCppOutputFormatter()
    
    def test_format_allocation_basic(self):
        """Test basic allocation formatting."""
        result = AllocationResult()
        
        # Create simple tensor mappings
        result.tensor_gpu_mapping = {
            "blk.0.attn_k.weight": 0,
            "blk.0.ffn_down.weight": 0,
            "blk.1.attn_v.weight": 1,
            "output.weight": 1
        }
        
        # Add GPU capacity for summary
        gpu0 = GPUCapacity(0, 4 * 1024**3, 80.0)
        gpu0.allocated_bytes = 1 * 1024**3
        gpu0.kv_cache_reserved_bytes = 512 * 1024**2
        gpu0.allocated_tensors = ["blk.0.attn_k.weight", "blk.0.ffn_down.weight"]
        gpu0.allocated_blocks = {0}
        
        gpu1 = GPUCapacity(1, 4 * 1024**3, 80.0)
        gpu1.allocated_bytes = 800 * 1024**2
        gpu1.kv_cache_reserved_bytes = 512 * 1024**2
        gpu1.allocated_tensors = ["blk.1.attn_v.weight", "output.weight"]
        gpu1.allocated_blocks = {1, None}
        
        result.gpu_allocations[0] = gpu0
        result.gpu_allocations[1] = gpu1
        
        output = self.formatter.format_allocation(result)
        
        # Verify llama.cpp flags are generated
        assert "# llama.cpp Tensor Override Flags" in output
        # Block 0 tensors should be collated to a single prefix
        assert '-ot "^blk\\.0\\..*=CUDA0"' in output
        # Block 1 has only one tensor in this test; collated prefix is acceptable
        assert '-ot "^blk\\.1\\..*=CUDA1"' in output
        assert '-ot "^output\\.weight$=CUDA1"' in output
        
        # Verify summary is included as comments
        assert "# Allocation Summary" in output
        assert "# Total tensors: 4" in output
        assert "# GPU 0:" in output
        assert "of usable" in output and "of total" in output
    
    def test_format_allocation_empty(self):
        """Test formatting with no tensor allocations."""
        result = AllocationResult()
        
        output = self.formatter.format_allocation(result)
        
        # Should have summary but no tensor flags
        assert "# Allocation Summary" in output
        assert "# Total tensors: 0" in output
        assert "llama.cpp Tensor Override Flags" not in output
        assert "-ot" not in output
    
    def test_format_allocation_with_warnings(self):
        """Test formatting with warnings."""
        result = AllocationResult()
        result.tensor_gpu_mapping = {"test.weight": 0}
        result.add_warning("Test warning message")
        
        gpu0 = GPUCapacity(0, 4 * 1024**3, 80.0)
        result.gpu_allocations[0] = gpu0
        
        output = self.formatter.format_allocation(result)
        
        # Verify warnings are included as comments
        assert "# Warnings:" in output
        assert "# ⚠️  Test warning message" in output
    
    def test_format_allocation_multiple_gpus(self):
        """Test formatting with multiple GPUs."""
        result = AllocationResult()
        
        # Distribute tensors across 3 GPUs
        result.tensor_gpu_mapping = {
            "blk.0.attn.weight": 0,
            "blk.1.attn.weight": 1, 
            "blk.2.attn.weight": 2,
            "blk.0.ffn.weight": 0,
            "blk.1.ffn.weight": 1,
            "blk.2.ffn.weight": 2
        }
        
        # Add GPU capacities
        for gpu_id in range(3):
            gpu = GPUCapacity(gpu_id, 2 * 1024**3, 90.0)
            result.gpu_allocations[gpu_id] = gpu
        
        output = self.formatter.format_allocation(result)
        
        # Verify all GPUs have corresponding CUDA device assignments
        assert "CUDA0" in output
        assert "CUDA1" in output
        assert "CUDA2" in output
        
        # Verify block-level collation
        assert '-ot "^blk\\.0\\..*=CUDA0"' in output
        assert '-ot "^blk\\.1\\..*=CUDA1"' in output
        assert '-ot "^blk\\.2\\..*=CUDA2"' in output
    
    def test_escape_tensor_name(self):
        """Test tensor name escaping for regex safety."""
        # Test escaping of common special characters
        test_cases = [
            ("blk.0.attn.weight", "blk\\.0\\.attn\\.weight"),
            ("model.layers[0].weight", "model\\.layers\\[0\\]\\.weight"),
            ("transformer.h.0.attn.c_attn.weight", "transformer\\.h\\.0\\.attn\\.c_attn\\.weight"),
            ("ffn_gate(bias)", "ffn_gate\\(bias\\)"),
            ("weight+bias", "weight\\+bias"),
            ("expert*gate", "expert\\*gate"),
            ("query?key", "query\\?key"),
            ("attention{head}", "attention\\{head\\}"),
            ("layer|norm", "layer\\|norm")
        ]
        
        for input_name, expected_escaped in test_cases:
            escaped = self.formatter._escape_tensor_name(input_name)
            assert escaped == expected_escaped, f"Failed for {input_name}: got {escaped}, expected {expected_escaped}"
    
    def test_generate_tensor_patterns(self):
        """Test tensor pattern generation."""
        tensor_names = [
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight", 
            "blk.1.ffn_down.weight"
        ]
        
        patterns = self.formatter._generate_tensor_patterns(tensor_names)
        
        # Should generate exact match patterns for each tensor
        expected_patterns = [
            "^blk\\.0\\.attn_k\\.weight$",
            "^blk\\.0\\.attn_v\\.weight$",
            "^blk\\.1\\.ffn_down\\.weight$"
        ]
        
        assert len(patterns) == len(expected_patterns)
        for pattern in expected_patterns:
            assert pattern in patterns
    
    def test_generate_tensor_patterns_empty(self):
        """Test pattern generation with empty input."""
        patterns = self.formatter._generate_tensor_patterns([])
        assert patterns == []
    
    def test_format_allocation_special_characters(self):
        """Test formatting with tensor names containing special characters."""
        result = AllocationResult()
        
        # Test tensor names with various special characters
        result.tensor_gpu_mapping = {
            "model.layers.0.self_attn.q_proj.weight": 0,
            "transformer.h.0.attn.c_attn.weight": 0,
            "layers[0].feed_forward.w1.weight": 1,
            "blk.0.ffn_gate(bias)": 1
        }
        
        # Add basic GPU capacity
        for gpu_id in range(2):
            gpu = GPUCapacity(gpu_id, 4 * 1024**3, 90.0)
            result.gpu_allocations[gpu_id] = gpu
        
        output = self.formatter.format_allocation(result)
        
        # Verify special characters are properly escaped
        assert '-ot "^model\\.layers\\.0\\.self_attn\\.q_proj\\.weight$=CUDA0"' in output
        assert '-ot "^transformer\\.h\\.0\\.attn\\.c_attn\\.weight$=CUDA0"' in output
        assert '-ot "^layers\\[0\\]\\.feed_forward\\.w1\\.weight$=CUDA1"' in output
        # blk.0 is collated to a prefix when all blk.0 tensors are on the same device
        assert '-ot "^blk\\.0\\..*=CUDA1"' in output
    
    def test_format_allocation_sorted_output(self):
        """Test that output is consistently sorted."""
        result = AllocationResult()
        
        # Add tensors in random order
        result.tensor_gpu_mapping = {
            "zzz.weight": 1,
            "aaa.weight": 0,
            "mmm.weight": 1,
            "bbb.weight": 0
        }
        
        # Add GPU capacities
        for gpu_id in range(2):
            gpu = GPUCapacity(gpu_id, 4 * 1024**3, 90.0)
            result.gpu_allocations[gpu_id] = gpu
        
        output = self.formatter.format_allocation(result)
        lines = output.split('\n')
        
        # Find override flag lines
        override_lines = [line for line in lines if line.startswith('-ot')]
        
        # Should be sorted by GPU first, then by tensor name
        expected_order = [
            '-ot "^aaa\\.weight$=CUDA0"',
            '-ot "^bbb\\.weight$=CUDA0"',
            '-ot "^mmm\\.weight$=CUDA1"',
            '-ot "^zzz\\.weight$=CUDA1"'
        ]
        
        assert override_lines == expected_order
    
    def test_format_allocation_comparison_with_generic(self):
        """Test that LlamaCpp formatter includes similar information to Generic formatter."""
        from gguf_tensor_overrider_py.core import GenericOutputFormatter
        
        result = AllocationResult()
        result.tensor_gpu_mapping = {
            "tensor1": 0,
            "tensor2": 1
        }
        
        # Add GPU capacities with some utilization
        gpu0 = GPUCapacity(0, 4 * 1024**3, 80.0)
        gpu0.allocated_bytes = 1 * 1024**3
        gpu0.kv_cache_reserved_bytes = 512 * 1024**2
        
        gpu1 = GPUCapacity(1, 8 * 1024**3, 90.0)
        gpu1.allocated_bytes = 2 * 1024**3
        gpu1.kv_cache_reserved_bytes = 1 * 1024**3
        
        result.gpu_allocations[0] = gpu0
        result.gpu_allocations[1] = gpu1
        
        # Format with both formatters
        generic_formatter = GenericOutputFormatter()
        llama_formatter = LlamaCppOutputFormatter()
        
        generic_output = generic_formatter.format_allocation(result)
        llama_output = llama_formatter.format_allocation(result)
        
        # Both should include allocation summary information (though formatted differently)
        assert "Total tensors: 2" in generic_output
        assert "# Total tensors: 2" in llama_output
        
        assert "GPU 0:" in generic_output
        assert "# GPU 0:" in llama_output
        
        assert "GPU 1:" in generic_output  
        assert "# GPU 1:" in llama_output
        
        # LlamaCpp should have additional override flags
        assert "-ot" in llama_output
        assert "-ot" not in generic_output
        
        # Generic should have tensor mappings
        assert "tensor1:gpu_0" in generic_output
        assert "tensor1:gpu_0" not in llama_output

    def test_natural_numeric_sorting(self):
        """Ensure -ot flags sort blocks numerically (1,2,19 rather than 1,19,2)."""
        result = AllocationResult()
        # Same GPU, different block numbers; include a global to ensure exact-match coexists
        result.tensor_gpu_mapping = {
            "blk.1.attn.weight": 0,
            "blk.19.ffn.weight": 0,
            "blk.2.attn.weight": 0,
            "output.weight": 0,
        }
        # One GPU capacity is sufficient
        result.gpu_allocations[0] = GPUCapacity(0, 8 * 1024**3, 90.0)
        output = self.formatter.format_allocation(result)
        # Collect -ot lines for CUDA0
        lines = [line for line in output.splitlines() if line.startswith('-ot') and 'CUDA0' in line]
        # Expected natural order: blk.1, blk.2, blk.19 prefixes then output.weight (exact)
        # Because blocks are collated, we check for their order in lines list
        expected_order_substrings = [
            '^blk\\.1\\..*=CUDA0',
            '^blk\\.2\\..*=CUDA0',
            '^blk\\.19\\..*=CUDA0',
        ]
        # Extract only lines that contain a blk prefix for ordering check
        blk_lines = [line for line in lines if 'blk' in line]
        # Ensure at least 3 blk lines exist
        assert len(blk_lines) >= 3
        # Verify order
        for i, sub in enumerate(expected_order_substrings):
            assert sub in blk_lines[i]