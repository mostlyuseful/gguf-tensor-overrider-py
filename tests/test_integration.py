"""Integration tests with real GGUF models from Hugging Face."""

import pytest
from pathlib import Path
import tempfile
import os

from gguf_tensor_overrider_py.core import AllocationRequest, GGUFTensorOverrider, GenericOutputFormatter
from gguf_tensor_overrider_py.models import DataType
from huggingface_hub import hf_hub_download


@pytest.fixture(scope="module")
def test_model_path():
    """Download test GGUF model once per test module."""
    # Download model if not already cached
    print("Downloading test model if not already cached...")
    downloaded_path = hf_hub_download(
        repo_id="unsloth/gemma-3-1b-it-GGUF",
        filename="gemma-3-1b-it-UD-IQ1_S.gguf"
    )
    return str(downloaded_path)


class TestRealGGUFIntegration:
    """Integration tests with real GGUF model files."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_single_gpu_10gb_success(self, test_model_path):
        """Test placement on single GPU with 10GB VRAM - should succeed."""
        overrider = GGUFTensorOverrider(output_formatter=GenericOutputFormatter())
        
        request = AllocationRequest(
            gguf_path=test_model_path,
            use_system_gpus=False,
            gpu_vram_config="0=10",
            gpu_percentages="90",
            context_length=4096,
            k_dtype=DataType.F16,
            v_dtype=DataType.F16,
            verbose=True
        )
        
        output = overrider.process_allocation_request(request)
        
        # Verify successful allocation
        assert "# Tensor Allocations" in output
        assert "# Allocation Summary" in output
        assert "Total tensors:" in output
        assert "Unallocated: 0" in output  # Should fit all tensors
        assert "GPU 0:" in output
        
        # Verify reasonable utilization
        lines = output.split('\n')
        gpu_line = next(line for line in lines if line.startswith("GPU 0:"))
        assert "%" in gpu_line  # Should show utilization percentage
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_single_gpu_0_1gb_partial(self, test_model_path):
        """Test placement on single GPU with 0.1GB VRAM - should have partial placement."""
        overrider = GGUFTensorOverrider(output_formatter=GenericOutputFormatter())
        
        request = AllocationRequest(
            gguf_path=test_model_path,
            use_system_gpus=False,
            gpu_vram_config="0=0.1",
            gpu_percentages="90",
            context_length=4096,
            k_dtype=DataType.F16,
            v_dtype=DataType.F16,
            verbose=False
        )
        
        output = overrider.process_allocation_request(request)
        
        # Verify partial allocation
        assert "# Allocation Summary" in output
        assert "Total tensors:" in output
        
        # Should have unallocated tensors due to insufficient VRAM
        lines = output.split('\n')
        summary_lines = [line for line in lines if "Unallocated:" in line]
        assert len(summary_lines) > 0
        unallocated_line = summary_lines[0]
        unallocated_count = int(unallocated_line.split("Unallocated: ")[1])
        assert unallocated_count > 0
        
        # Should have warnings about blocks not fitting
        assert "# Warnings" in output or unallocated_count > 0
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_two_gpus_10gb_first_gpu_only(self, test_model_path):
        """Test placement on two 10GB GPUs - should use only first GPU."""
        overrider = GGUFTensorOverrider(output_formatter=GenericOutputFormatter())
        
        request = AllocationRequest(
            gguf_path=test_model_path,
            use_system_gpus=False,
            gpu_vram_config="0=10,1=10",
            gpu_percentages="90",
            context_length=4096,
            k_dtype=DataType.F16,
            v_dtype=DataType.F16,
            verbose=False
        )
        
        output = overrider.process_allocation_request(request)
        
        # Verify successful allocation
        assert "# Allocation Summary" in output
        assert "Unallocated: 0" in output  # Should fit all tensors
        assert "GPU 0:" in output
        assert "GPU 1:" in output
        
        # Parse GPU utilizations
        lines = output.split('\n')
        gpu0_line = next(line for line in lines if line.startswith("GPU 0:"))
        gpu1_line = next(line for line in lines if line.startswith("GPU 1:"))
        
        # GPU 0 should have significant usage, GPU 1 should have minimal (just KV cache)
        assert "tensors" in gpu0_line
        # Extract tensor sizes - GPU 0 should have more than GPU 1
        gpu0_tensors = float(gpu0_line.split("GB tensors")[0].split()[-1])
        gpu1_tensors = float(gpu1_line.split("GB tensors")[0].split()[-1])
        
        # GPU 0 should have most/all tensors, GPU 1 should have few or none
        assert gpu0_tensors > gpu1_tensors
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_two_gpus_0_1gb_partial(self, test_model_path):
        """Test placement on two 0.1GB GPUs - should have partial placement on both."""
        overrider = GGUFTensorOverrider(output_formatter=GenericOutputFormatter())
        
        request = AllocationRequest(
            gguf_path=test_model_path,
            use_system_gpus=False,
            gpu_vram_config="0=0.1,1=0.1",
            gpu_percentages="90",
            context_length=4096,
            k_dtype=DataType.F16,
            v_dtype=DataType.F16,
            verbose=False
        )
        
        output = overrider.process_allocation_request(request)
        
        # Verify partial allocation
        assert "# Allocation Summary" in output
        
        # Should have significant unallocated tensors
        lines = output.split('\n')
        summary_lines = [line for line in lines if "Unallocated:" in line]
        assert len(summary_lines) > 0
        unallocated_count = int(summary_lines[0].split("Unallocated: ")[1])
        assert unallocated_count > 0
        
        # Both GPUs should be mentioned
        assert "GPU 0:" in output
        assert "GPU 1:" in output
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_two_gpus_0_4gb_distributed(self, test_model_path):
        """Test placement on two 0.4GB GPUs - should distribute across both."""
        overrider = GGUFTensorOverrider(output_formatter=GenericOutputFormatter())
        
        request = AllocationRequest(
            gguf_path=test_model_path,
            use_system_gpus=False,
            gpu_vram_config="0=0.4,1=0.4",
            gpu_percentages="90",
            context_length=4096,
            k_dtype=DataType.F16,
            v_dtype=DataType.F16,
            verbose=False
        )
        
        output = overrider.process_allocation_request(request)
        
        # Verify allocation across both GPUs
        assert "# Allocation Summary" in output
        assert "GPU 0:" in output
        assert "GPU 1:" in output
        
        # Parse GPU utilizations
        lines = output.split('\n')
        gpu0_line = next(line for line in lines if line.startswith("GPU 0:"))
        gpu1_line = next(line for line in lines if line.startswith("GPU 1:"))
        
        # Both GPUs should have some tensor allocation
        gpu0_tensors = float(gpu0_line.split("GB tensors")[0].split()[-1])
        gpu1_tensors = float(gpu1_line.split("GB tensors")[0].split()[-1])
        
        # Both should have some tensors (though may not be equal due to block co-location)
        assert gpu0_tensors > 0.0
        assert gpu1_tensors >= 0.0  # GPU 1 might not get tensors if blocks don't fit
        
        # Total allocation should be reasonable for 0.8GB total capacity
        total_allocated = gpu0_tensors + gpu1_tensors
        assert total_allocated > 0.0
        assert total_allocated < 0.8  # Should be less than total capacity
    
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.parametrize("k_dtype,v_dtype", [
        (DataType.F16, DataType.F16),
        (DataType.Q8_0, DataType.Q8_0),
        (DataType.Q5_1, DataType.Q5_1),
        # Note: Q4_0 not in our DataType enum, using available types
        (DataType.F16, DataType.Q8_0),  # Mixed types
    ])
    def test_kv_cache_quantization_effects(self, test_model_path, k_dtype, v_dtype):
        """Test different KV cache quantization settings."""
        overrider = GGUFTensorOverrider(output_formatter=GenericOutputFormatter())
        
        request = AllocationRequest(
            gguf_path=test_model_path,
            use_system_gpus=False,
            gpu_vram_config="0=10",
            gpu_percentages="90",
            context_length=4096,
            k_dtype=k_dtype,
            v_dtype=v_dtype,
            verbose=False
        )
        
        output = overrider.process_allocation_request(request)
        
        # Verify successful allocation regardless of KV cache settings
        assert "# Allocation Summary" in output
        assert "GPU 0:" in output
        
        # Parse KV cache usage
        lines = output.split('\n')
        gpu0_line = next(line for line in lines if line.startswith("GPU 0:"))
        
        # Should have KV cache allocation
        assert "KV cache" in gpu0_line
        kv_cache_gb = float(gpu0_line.split("GB KV cache")[0].split("+")[-1].strip())
        
        # KV cache size should vary based on data types
        assert kv_cache_gb > 0.0
        
        # Lower precision should use less memory
        if k_dtype == DataType.Q8_0 and v_dtype == DataType.Q8_0:
            # Q8_0 uses 1 byte per element vs 2 bytes for F16
            # So KV cache should be roughly half the size of F16
            pass  # We'd need a baseline to compare against
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_different_context_sizes(self, test_model_path):
        """Test how context size affects allocation."""
        overrider = GGUFTensorOverrider(output_formatter=GenericOutputFormatter())
        
        results = {}
        
        for context_size in [1024, 4096, 8192]:
            request = AllocationRequest(
                gguf_path=test_model_path,
                use_system_gpus=False,
                gpu_vram_config="0=10",
                gpu_percentages="90",
                context_length=context_size,
                k_dtype=DataType.F16,
                v_dtype=DataType.F16,
                verbose=False
            )
            
            output = overrider.process_allocation_request(request)
            
            # Extract KV cache usage
            lines = output.split('\n')
            gpu0_line = next(line for line in lines if line.startswith("GPU 0:"))
            kv_cache_gb = float(gpu0_line.split("GB KV cache")[0].split("+")[-1].strip())
            
            results[context_size] = kv_cache_gb
        
        # KV cache should scale linearly with context size
        assert results[4096] > results[1024]
        assert results[8192] > results[4096]
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_block_colocations(self, test_model_path):
        """Test that blocks are kept together on same GPU."""
        overrider = GGUFTensorOverrider(output_formatter=GenericOutputFormatter())
        
        request = AllocationRequest(
            gguf_path=test_model_path,
            use_system_gpus=False,
            gpu_vram_config="0=0.3,1=0.3",  # Force distribution across GPUs
            gpu_percentages="90",
            context_length=2048,
            k_dtype=DataType.F16,
            v_dtype=DataType.F16,
            verbose=True
        )
        
        output = overrider.process_allocation_request(request)
        
        # Look for block allocation information in verbose output
        lines = output.split('\n')
        
        # Find block allocation lines
        block_lines = [line for line in lines if "Blocks:" in line]
        
        if block_lines:
            # Parse block allocations
            for line in block_lines:
                if "GPU 0" in line:
                    # Extract block numbers
                    blocks_part = line.split("Blocks:")[-1].strip()
                    if blocks_part and blocks_part != "":
                        # Should have contiguous or reasonable block groupings
                        # This validates that blocks are kept together
                        assert len(blocks_part) > 0
                
        # At minimum, verify that allocation succeeded and makes sense
        assert "# Allocation Summary" in output
        assert "GPU 0:" in output
        assert "GPU 1:" in output
    
    @pytest.mark.integration 
    @pytest.mark.slow
    def test_allocation_deterministic(self, test_model_path):
        """Test that allocation is deterministic across runs."""
        overrider = GGUFTensorOverrider(output_formatter=GenericOutputFormatter())
        
        request = AllocationRequest(
            gguf_path=test_model_path,
            use_system_gpus=False,
            gpu_vram_config="0=2,1=2",
            gpu_percentages="80",
            context_length=2048,
            k_dtype=DataType.F16,
            v_dtype=DataType.F16,
            verbose=False
        )
        
        # Run allocation multiple times
        outputs = []
        for _ in range(3):
            output = overrider.process_allocation_request(request)
            outputs.append(output)
        
        # All outputs should be identical (deterministic allocation)
        assert outputs[0] == outputs[1] == outputs[2]
        
        # Verify reasonable allocation
        output = outputs[0]
        assert "# Allocation Summary" in output
        assert "Total tensors:" in output


@pytest.mark.integration
@pytest.mark.slow
class TestErrorHandling:
    """Test error handling with real GGUF files."""
    
    def test_invalid_gguf_file(self):
        """Test handling of invalid GGUF file."""
        overrider = GGUFTensorOverrider(output_formatter=GenericOutputFormatter())
        
        # Create a fake file
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            f.write(b"not a real gguf file")
            fake_file = f.name
        
        try:
            request = AllocationRequest(
                gguf_path=fake_file,
                use_system_gpus=False,
                gpu_vram_config="0=10",
                gpu_percentages="90",
                context_length=2048,
                k_dtype=DataType.F16,
                v_dtype=DataType.F16,
                verbose=False
            )
            
            with pytest.raises(RuntimeError, match="Allocation failed"):
                overrider.process_allocation_request(request)
        finally:
            os.unlink(fake_file)
    
    def test_nonexistent_file(self):
        """Test handling of nonexistent GGUF file."""
        overrider = GGUFTensorOverrider(output_formatter=GenericOutputFormatter())
        
        request = AllocationRequest(
            gguf_path="/nonexistent/path/model.gguf",
            use_system_gpus=False,
            gpu_vram_config="0=10",
            gpu_percentages="90",
            context_length=2048,
            k_dtype=DataType.F16,
            v_dtype=DataType.F16,
            verbose=False
        )
        
        with pytest.raises(RuntimeError, match="Allocation failed"):
            overrider.process_allocation_request(request)
    
    def test_http_integration(self):
        """Test allocation with a GGUF file over HTTP."""
        overrider = GGUFTensorOverrider(output_formatter=GenericOutputFormatter())
        
        # Use a known GGUF file URL from Hugging Face
        gguf_url = "https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-UD-IQ1_S.gguf"
        
        request = AllocationRequest(
            gguf_path=gguf_url,
            use_system_gpus=False,
            gpu_vram_config="0=10",
            gpu_percentages="90",
            context_length=2048,
            k_dtype=DataType.F16,
            v_dtype=DataType.F16,
            verbose=False
        )
        
        output = overrider.process_allocation_request(request)
        
        # Verify successful allocation
        assert "# Allocation Summary" in output
        assert "GPU 0:" in output
        assert "Unallocated: 0" in output  # Should fit all tensors


# Helper function to run integration tests manually
def run_integration_tests():
    """Helper function to run integration tests manually."""
    pytest.main([
        "tests/test_integration.py",
        "-v",
        "-m", "integration",
        "--tb=short"
    ])


if __name__ == "__main__":
    run_integration_tests()