# GGUF Tensor Overrider

This is a reimplementation of the Javascript project https://github.com/k-koehler/gguf-tensor-overrider in Python. It does not reuse any code from the original project and aims to provide similar functionality with a different implementation approach.

## Goal

The goal of this project is to provide a simple way to generate llama.cpp / ik_llama.cpp tensor override CLI flags. The tensor overrides are used to place specific tensors in the VRAM of a particular NVIDIA GPU instead of CPU RAM, which should improve performance when running large models.

## Invocation

```shell
# This will generate tensor-override flags using your actually installed GPUs
uvx --from=https://github.com/mostlyuseful/gguf-tensor-overrider-py.git --gguf URL/file path to gguf model --use-system-gpus --context 2048 -ctk q8_0 -ctv q5_1 --gpu-percentage 0=20,1=80,90

# This will generate tensor-override flags using 2 hypothetical GPUs with 6GB and 10GB of VRAM
uvx --from=https://github.com/mostlyuseful/gguf-tensor-overrider-py.git --gguf URL/file path to gguf model --context 2048 -ctk q8_0 -ctv q5_1 --gpu-vram 0=6,1=10 --gpu-percentage 90
```

## Options

- `--gguf <URL or file path>`: URL or file path to the GGUF model.
- `--use-system-gpus`: Use the GPUs installed on the system. This option is mutually exclusive with `--gpu-vram`.
- `--gpu-vram <index=GB,...>`: Comma-separated list of hypothetical GPUs with their VRAM sizes in GB. This option is mutually exclusive with `--use-system-gpus`.
- `--context <number>`: Context size for the model (default: 2048).
- `-ctk <type>`: KV cache K data type (default: f16).
- `-ctv <type>`: KV cache V data type (default: f16).
- `--gpu-percentage <index=percentage,...|percentage>`: Comma-separated list of GPU indices with their respective percentages of VRAM to use for tensor overrides. If a single percentage is provided, it applies to all GPUs not overridden by index. Defaults to 90% if not specified (corresponds to `--gpu-percentage 90`).
- `--help`: Show help message and exit.
- `--version`: Show version information and exit.
- `--verbose`: Enable verbose output for debugging purposes.
