"""Shared helpers for CLI commands."""

from pathlib import Path

import typer
from rich.console import Console

from finetuning.core.config import AppConfig, load_config
from finetuning.core.exceptions import ConfigurationError

console = Console()

CONFIG_DIR_OPTION = typer.Option(
    Path("configs"), "--config-dir", help="Directory with Hydra config groups."
)
OVERRIDES_ARGUMENT = typer.Argument(None, help="Hydra overrides, e.g. lora.r=8 model=qwen3-0.6b")


def load_cli_config(config_dir: Path, overrides: list[str] | None) -> AppConfig:
    try:
        return load_config(config_dir=config_dir, overrides=overrides)
    except ConfigurationError as error:
        console.print(f"[red]Configuration error:[/red] {error}")
        raise typer.Exit(code=2) from error
