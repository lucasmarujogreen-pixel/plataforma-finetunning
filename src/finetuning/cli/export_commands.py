"""CLI commands for exporting trained models."""

from pathlib import Path

import typer
from rich.table import Table

from finetuning.application.export_model import ExportModel, ExportResult
from finetuning.cli.common import console
from finetuning.core.enums import ExportFormat
from finetuning.core.exceptions import PlatformError
from finetuning.exporters.merge import merge_lora_adapter
from finetuning.infrastructure.experiment_manager import load_run_config

RUN_DIR_OPTION = typer.Option(..., "--run", help="Run directory to export.")
FORMATS_OPTION = typer.Option(
    None,
    "--format",
    help="Export formats (default from config): lora_adapter, " "merged_safetensors, gguf.",
)


def _print_result(result: ExportResult) -> None:
    table = Table(title=f"Export: {result.run_name}")
    table.add_column("Artifact")
    table.add_column("Path")
    for name, path in result.artifacts.items():
        table.add_row(name, path)
    console.print(table)


def export(
    run_dir: Path = RUN_DIR_OPTION,
    formats: list[ExportFormat] = FORMATS_OPTION,
) -> None:
    """Export a trained run: LoRA adapter, merged safetensors, GGUF and Modelfile."""
    try:
        result = ExportModel().execute(run_dir, list(formats) if formats else None)
    except PlatformError as error:
        console.print(f"[red]Export error:[/red] {error}")
        raise typer.Exit(code=1) from error
    _print_result(result)


def merge_lora(run_dir: Path = RUN_DIR_OPTION) -> None:
    """Merge the run's LoRA adapter into the base model."""
    try:
        config = load_run_config(run_dir)
        output_dir = merge_lora_adapter(config, run_dir / "model", run_dir / "exported" / "merged")
    except PlatformError as error:
        console.print(f"[red]Merge error:[/red] {error}")
        raise typer.Exit(code=1) from error
    console.print(f"Merged model: [green]{output_dir}[/green]")
