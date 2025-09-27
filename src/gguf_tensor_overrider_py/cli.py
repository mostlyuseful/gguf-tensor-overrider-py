"""Command-line interface for GGUF Tensor Overrider."""

from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from gguf_tensor_overrider_py.core import (
    AllocationRequest,
    GGUFTensorOverrider,
    LinearTensorAllocator,
    PriorityTensorAllocator,
)
from gguf_tensor_overrider_py.models import DataType

app = typer.Typer(
    name="gguf-tensor-overrider-py",
    help="Generate tensor override flags for llama.cpp and ik_llama.cpp",
    no_args_is_help=True,
)


def parse_data_type(value: str) -> DataType:
    """Parse data type string to DataType enum."""
    try:
        return DataType(value.lower())
    except ValueError:
        valid_types = [dt.value for dt in DataType]
        raise typer.BadParameter(
            f"Invalid data type '{value}'. Valid types: {', '.join(valid_types)}"
        )


def validate_gpu_percentage(value: str) -> str:
    """Validate GPU percentage configuration string."""
    try:
        for spec in value.split(","):
            spec = spec.strip()
            if "=" in spec:
                # Index-specific: "0=80"
                index_str, percent_str = spec.split("=", 1)
                int(index_str)  # Validate index is integer
                percent = float(percent_str)
                if not (1 <= percent <= 100):
                    raise ValueError(f"Percentage must be 1-100, got {percent}")
            else:
                # Default percentage: "90"
                percent = float(spec)
                if not (1 <= percent <= 100):
                    raise ValueError(f"Percentage must be 1-100, got {percent}")
        return value
    except ValueError as e:
        raise typer.BadParameter(f"Invalid GPU percentage format: {e}")


class AllocatorStrategy(str, Enum):
    LINEAR = "linear"
    PRIORITY = "priority"


def get_allocator(strat: AllocatorStrategy):
    match strat:
        case AllocatorStrategy.LINEAR:
            return LinearTensorAllocator()
        case AllocatorStrategy.PRIORITY:
            return PriorityTensorAllocator()
        case _:
            raise ValueError(f"Unknown allocator strategy: {strat}")


def generate_report(placements, filepath: str) -> None:
    """
    Generate a Markdown allocation report.

    The function is defensive: it attempts to infer common attribute/key names:
      name: name / tensor_name
      device: device / gpu / device_index / gpu_index
      size bytes: nbytes / size_bytes / size / bytes
    """
    try:
        lines: list[str] = []
        lines.append("# Tensor Allocation Report\n")
        if placements is None:
            lines.append("_No placement data available._\n")
        else:
            total = len(placements)
            lines.append(f"**Total tensors:** {total}\n")
            device_summary = {}
            device_mem = {}

            def _extract(obj, candidates):
                # dict first
                if isinstance(obj, dict):
                    for c in candidates:
                        if c in obj:
                            return obj[c]
                # attribute lookup
                for c in candidates:
                    if hasattr(obj, c):
                        return getattr(obj, c)
                return None

            table_rows = []
            for p in placements:
                name = _extract(p, ("name", "tensor_name", "key", "id")) or "?"
                device = _extract(p, ("device", "gpu", "device_index", "gpu_index", "placement")) 
                size_bytes = _extract(p, ("nbytes", "size_bytes", "bytes", "size"))
                # Normalize device
                if isinstance(device, (tuple, list)) and len(device) == 1:
                    device = device[0]
                if device is None:
                    device = "CPU?"
                device_summary[device] = device_summary.get(device, 0) + 1
                if isinstance(size_bytes, (int, float)):
                    device_mem[device] = device_mem.get(device, 0) + int(size_bytes)
                # Human readable size
                if isinstance(size_bytes, (int, float)):
                    mb = size_bytes / (1024**2)
                    size_disp = f"{mb:.2f} MB"
                else:
                    size_disp = "?"
                table_rows.append((str(name), str(device), size_disp))

            # Device summary
            lines.append("## Summary\n")
            lines.append("| Device | Tensor Count | Total Memory (MB) |")
            lines.append("|--------|--------------|-------------------|")
            for dev, count in sorted(device_summary.items(), key=lambda x: str(x[0])):
                mem_mb = ""
                if dev in device_mem:
                    mem_mb = f"{device_mem[dev]/(1024**2):.2f}"
                lines.append(f"| {dev} | {count} | {mem_mb} |")
            lines.append("")

            # Detailed table
            lines.append("## Tensor Placements\n")
            lines.append("| Tensor | Device | Size |")
            lines.append("|--------|--------|------|")
            for name, device, size_disp in table_rows:
                lines.append(f"| {name} | {device} | {size_disp} |")
            lines.append("")

        out_path = Path(filepath)
        out_path.write_text("\n".join(lines), encoding="utf-8")
    except Exception as e:
        # Best-effort fallback
        try:
            Path(filepath).write_text(f"# Tensor Allocation Report\nError: {e}\n", encoding="utf-8")
        except Exception:
            pass


@app.command()
def override(
    gguf: Annotated[str, typer.Argument(help="URL or file path to the GGUF model")],
    # GPU Configuration (mutually exclusive)
    use_system_gpus: Annotated[
        bool, typer.Option("--use-system-gpus", help="Use GPUs installed on the system")
    ] = False,
    gpu_vram: Annotated[
        Optional[str],
        typer.Option(
            "--gpu-vram",
            help="Comma-separated list of hypothetical GPUs (index=GB,...)",
        ),
    ] = None,
    # Model Parameters
    context: Annotated[
        int, typer.Option("--context", help="Context size for the model")
    ] = 2048,
    ctk: Annotated[str, typer.Option("-ctk", help="KV cache K data type")] = "f16",
    ctv: Annotated[str, typer.Option("-ctv", help="KV cache V data type")] = "f16",
    # GPU Allocation
    gpu_percentage: Annotated[
        str,
        typer.Option(
            "--gpu-percentage",
            help="VRAM usage limits per GPU",
            callback=validate_gpu_percentage,
        ),
    ] = "90",
    # Allocator Strategy
    allocator_strategy: Annotated[
        AllocatorStrategy,
        typer.Option(
            "--allocator",
            help="Tensor allocation strategy",
            case_sensitive=False,
        ),
    ] = AllocatorStrategy.PRIORITY,
    report: Annotated[
        bool,
        typer.Option(
            "--report", help="Generate a detailed allocation report in report.md"
        ),
    ] = False,
    # Utility Options
    verbose: Annotated[
        bool, typer.Option("--verbose", help="Enable verbose output for debugging")
    ] = False,
    version: Annotated[
        bool, typer.Option("--version", help="Show version information and exit")
    ] = False,
) -> None:
    """Generate tensor override flags for llama.cpp and ik_llama.cpp."""

    if verbose:
        import logging

        logging.basicConfig(level=logging.DEBUG, force=True)

    if version:
        typer.echo("gguf-tensor-overrider-py 0.1.0")
        raise typer.Exit()

    # Validate GPU configuration
    if use_system_gpus and gpu_vram:
        typer.echo("Error: Cannot use both --use-system-gpus and --gpu-vram", err=True)
        raise typer.Exit(1)

    if not use_system_gpus and not gpu_vram:
        typer.echo(
            "Error: Must specify either --use-system-gpus or --gpu-vram", err=True
        )
        raise typer.Exit(1)

    # Validate context size
    if context <= 0:
        typer.echo("Error: Context size must be positive", err=True)
        raise typer.Exit(1)

    # Parse data types
    try:
        k_dtype = parse_data_type(ctk)
        v_dtype = parse_data_type(ctv)
    except typer.BadParameter as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    # Validate GGUF path
    if not gguf.startswith(("http://", "https://")) and not Path(gguf).exists():
        typer.echo(f"Error: GGUF file not found: {gguf}", err=True)
        raise typer.Exit(1)

    # Create allocation request
    request = AllocationRequest(
        gguf_path=gguf,
        use_system_gpus=use_system_gpus,
        gpu_vram_config=gpu_vram,
        gpu_percentages=gpu_percentage,
        context_length=context,
        k_dtype=k_dtype,
        v_dtype=v_dtype,
        verbose=verbose,
    )

    # Process allocation
    allocator = get_allocator(allocator_strategy)
    try:
        overrider = GGUFTensorOverrider(allocator=allocator)
        placements, output = overrider.process_allocation_request(request)
        for line in output.splitlines():
            if line.strip().startswith("#"):
                typer.echo(line, err=True)
            else:
                typer.echo(line)
        if report:
            generate_report(placements, "report.md")
    except Exception as e:
        if verbose:
            import traceback

            traceback.print_exc()
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def check_gpus() -> None:
    """Check available GPUs on the system."""
    try:
        import pynvml

        pynvml.nvmlInit()

        device_count = pynvml.nvmlDeviceGetCount()
        typer.echo(f"Found {device_count} NVIDIA GPU(s):")

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_gb = mem_info.total / (1024**3)

            typer.echo(f"  GPU {i}: {name} ({vram_gb:.1f} GB VRAM)")

    except ImportError:
        typer.echo(
            "Error: pynvml not installed. Install with: pip install pynvml", err=True
        )
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error detecting GPUs: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
