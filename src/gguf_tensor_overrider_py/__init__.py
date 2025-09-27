"""GGUF Tensor Overrider - Generate GPU allocation plans for GGUF models."""

from gguf_tensor_overrider_py.cli import app
from gguf_tensor_overrider_py.core import GGUFTensorOverrider, AllocationRequest, GenericOutputFormatter, LlamaCppOutputFormatter
from gguf_tensor_overrider_py.models import (
    DataType, TensorPriority, TensorInfo, GPUCapacity, ModelMetadata,
    KVCacheConfig, BlockGroup, AllocationResult, GPUConfiguration
)

__version__ = "0.1.0"
__all__ = [
    "app",
    "GGUFTensorOverrider", 
    "AllocationRequest",
    "DataType",
    "TensorPriority", 
    "TensorInfo",
    "GPUCapacity",
    "ModelMetadata",
    "KVCacheConfig", 
    "BlockGroup",
    "AllocationResult",
    "GPUConfiguration",
]


def main():
    """Main entry point for the CLI."""
    app()