"""GGUF Tensor Overrider - Generate GPU allocation plans for GGUF models."""

from .cli import app
from .core import GGUFTensorOverrider, AllocationRequest, GenericOutputFormatter, LlamaCppOutputFormatter
from .models import (
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