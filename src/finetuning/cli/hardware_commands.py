"""CLI commands for hardware inspection."""

import platform

import typer
from rich.console import Console
from rich.table import Table

from finetuning import __version__
from finetuning.monitoring.hardware import HardwareProfile, detect_hardware

console = Console()


def _bool_label(value: bool) -> str:
    return "[green]yes[/green]" if value else "[red]no[/red]"


def _render_gpu_table(profile: HardwareProfile) -> Table:
    table = Table(title="GPUs")
    table.add_column("Index", justify="right")
    table.add_column("Name")
    table.add_column("VRAM (free/total MB)", justify="right")
    table.add_column("Compute capability", justify="center")
    for gpu in profile.gpus:
        table.add_row(
            str(gpu.index),
            gpu.name,
            f"{gpu.free_vram_mb} / {gpu.total_vram_mb}",
            f"{gpu.compute_capability[0]}.{gpu.compute_capability[1]}",
        )
    return table


def system_info() -> None:
    """Show platform, Python, RAM, CPU and CUDA capability summary."""
    profile = detect_hardware()
    table = Table(title=f"System info (finetuning {__version__})")
    table.add_column("Property")
    table.add_column("Value")
    table.add_row("OS", profile.os_description)
    table.add_row("Python", platform.python_version())
    table.add_row("CPU cores", str(profile.cpu_count))
    table.add_row("RAM (MB)", str(profile.total_ram_mb))
    table.add_row("CUDA available", _bool_label(profile.cuda_available))
    table.add_row("CUDA version", profile.cuda_version or "-")
    table.add_row("NVIDIA driver", profile.driver_version or "-")
    table.add_row("bf16 support", _bool_label(profile.supports_bf16))
    table.add_row("fp16 support", _bool_label(profile.supports_fp16))
    table.add_row("Flash Attention", _bool_label(profile.flash_attention_available))
    table.add_row("xFormers", _bool_label(profile.xformers_available))
    table.add_row("torch.compile", _bool_label(profile.torch_compile_supported))
    console.print(table)
    if profile.gpus:
        console.print(_render_gpu_table(profile))


def gpu_info() -> None:
    """Show detected GPUs with VRAM and compute capability."""
    profile = detect_hardware()
    if not profile.gpus:
        console.print("[yellow]No CUDA GPU detected. CPU mode is for debugging only.[/yellow]")
        raise typer.Exit(code=1)
    console.print(_render_gpu_table(profile))
