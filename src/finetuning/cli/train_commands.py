"""CLI commands for training and resuming experiments."""

from pathlib import Path

import typer
from rich.table import Table

from finetuning.application.resume_training import ResumeTraining
from finetuning.application.train_model import TrainModel, TrainModelResult
from finetuning.cli.common import CONFIG_DIR_OPTION, OVERRIDES_ARGUMENT, console, load_cli_config
from finetuning.core.exceptions import PlatformError

RUN_DIR_OPTION = typer.Option(..., "--run", help="Run directory to resume from.")


def _print_result(result: TrainModelResult) -> None:
    table = Table(title="Training result")
    table.add_column("Property")
    table.add_column("Value")
    table.add_row("Run", str(result.run_dir))
    table.add_row("Experiment ID", result.experiment_id)
    table.add_row("Total time (s)", f"{result.total_seconds:.1f}")
    for key in ("train_loss", "eval_loss", "train_samples_per_second"):
        if key in result.metrics:
            table.add_row(key, f"{result.metrics[key]:.4f}")
    console.print(table)


def train(
    overrides: list[str] = OVERRIDES_ARGUMENT,
    config_dir: Path = CONFIG_DIR_OPTION,
) -> None:
    """Run a fine-tuning experiment with the composed configuration."""
    config = load_cli_config(config_dir, overrides)
    try:
        result = TrainModel().execute(config)
    except PlatformError as error:
        console.print(f"[red]Training error:[/red] {error}")
        raise typer.Exit(code=1) from error
    _print_result(result)


def resume(
    run_dir: Path = RUN_DIR_OPTION,
) -> None:
    """Resume training from the last checkpoint of an existing run."""
    try:
        result = ResumeTraining().execute(run_dir)
    except PlatformError as error:
        console.print(f"[red]Resume error:[/red] {error}")
        raise typer.Exit(code=1) from error
    _print_result(result)
