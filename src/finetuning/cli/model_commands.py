"""CLI commands for model management."""

import shutil
from pathlib import Path

import typer
from rich.table import Table

from finetuning.application.download_model import DownloadModel
from finetuning.cli.common import CONFIG_DIR_OPTION, OVERRIDES_ARGUMENT, console, load_cli_config
from finetuning.infrastructure.huggingface import HuggingFaceModelStore


def download_model(
    overrides: list[str] = OVERRIDES_ARGUMENT,
    config_dir: Path = CONFIG_DIR_OPTION,
    skip_hash: bool = typer.Option(False, "--skip-hash", help="Skip SHA256 fingerprinting."),
) -> None:
    """Download the configured base model from the Hugging Face Hub."""
    config = load_cli_config(config_dir, overrides)
    use_case = DownloadModel(HuggingFaceModelStore(config.model.cache_dir))
    result = use_case.execute(config.model, compute_hash=not skip_hash)

    table = Table(title="Model snapshot")
    table.add_column("Property")
    table.add_column("Value")
    table.add_row("Name", result.name)
    table.add_row("Revision", result.revision)
    table.add_row("Path", str(result.path))
    table.add_row("Files", str(result.file_count))
    table.add_row("Size (MB)", str(result.size_mb))
    table.add_row("SHA256", result.model_hash or "-")
    console.print(table)


def clean_cache(
    overrides: list[str] = OVERRIDES_ARGUMENT,
    config_dir: Path = CONFIG_DIR_OPTION,
    include_models: bool = typer.Option(
        False, "--include-models", help="Also remove downloaded model snapshots."
    ),
    yes: bool = typer.Option(False, "--yes", help="Do not ask for confirmation."),
) -> None:
    """Remove dataset caches (and optionally model snapshots)."""
    config = load_cli_config(config_dir, overrides)
    targets = [config.dataset.cache_dir, config.dataset.processed_dir]
    if include_models:
        targets.append(config.model.cache_dir)
    existing = [target for target in targets if target.exists()]
    if not existing:
        console.print("Nothing to clean.")
        return
    for target in existing:
        console.print(f"  - {target}")
    if not yes and not typer.confirm("Remove the directories above?"):
        raise typer.Exit(code=1)
    for target in existing:
        shutil.rmtree(target)
        console.print(f"[green]Removed[/green] {target}")
