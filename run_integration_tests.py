#!/usr/bin/env python3
"""
Script to run integration tests for GGUF Tensor Overrider.

This script downloads a real GGUF model and tests various GPU configurations.
"""

import sys
import subprocess
from pathlib import Path


def main():
    """Run integration tests with real GGUF models."""
    
    print("🧪 GGUF Tensor Overrider Integration Tests")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Error: Run this script from the project root directory")
        sys.exit(1)
    
    print("📋 This will:")
    print("  • Download a real GGUF model (gemma-3-1b-it-UD-IQ1_S.gguf)")
    print("  • Test various GPU configurations")
    print("  • Validate allocation algorithms with real data")
    print()
    
    print("🚀 Running integration tests...")
    print()
    
    # Run the integration tests
    cmd = [
        "uv", "run", "python", "-m", "pytest", 
        "tests/test_integration.py",
        "-v",
        "-m", "integration",
        "--tb=short"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print()
        print("✅ Integration tests completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print()
        print(f"❌ Integration tests failed with exit code {e.returncode}")
        print()
        print("💡 Tips:")
        print("  • Make sure you have internet access for model download")
        print("  • Check that huggingface_hub is installed: uv sync")
        print("  • Review test output above for specific errors")
        sys.exit(1)
    
    except KeyboardInterrupt:
        print()
        print("⏹️  Tests interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()