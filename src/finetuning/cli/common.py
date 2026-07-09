"""Shared helpers for CLI commands."""

from pathlib import Path
from typing import overload

import typer
from pydantic import BaseModel
from rich.console import Console

from finetuning.core.config import AppConfig, load_config
from finetuning.core.config.loader import DEFAULT_CONFIG_NAME
from finetuning.core.exceptions import ConfigurationError

console = Console()

CONFIG_DIR_OPTION = typer.Option(
    Path("configs"), "--config-dir", help="Directory with Hydra config groups."
)
OVERRIDES_ARGUMENT = typer.Argument(None, help="Hydra overrides, e.g. lora.r=8 model=qwen3-0.6b")


@overload
def load_cli_config(
    config_dir: Path, overrides: list[str] | None, config_name: str = DEFAULT_CONFIG_NAME
) -> AppConfig: ...
@overload
def load_cli_config[ConfigT: BaseModel](
    config_dir: Path,
    overrides: list[str] | None,
    config_name: str = DEFAULT_CONFIG_NAME,
    *,
    schema: type[ConfigT],
) -> ConfigT: ...
def load_cli_config(
    config_dir: Path,
    overrides: list[str] | None,
    config_name: str = DEFAULT_CONFIG_NAME,
    schema: type[BaseModel] = AppConfig,
) -> BaseModel:
    try:
        return load_config(
            config_dir=config_dir, config_name=config_name, overrides=overrides, schema=schema
        )
    except ConfigurationError as error:
        console.print(f"[red]Configuration error:[/red] {error}")
        raise typer.Exit(code=2) from error
