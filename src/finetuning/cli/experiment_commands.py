"""CLI commands for listing, inspecting and comparing experiments."""

from pathlib import Path

import typer
from rich.table import Table

from finetuning.application.compare_experiments import (
    CompareExperiments,
    load_evaluation_report,
    summarize_run,
)
from finetuning.cli.common import CONFIG_DIR_OPTION, console, load_cli_config
from finetuning.core.exceptions import PlatformError
from finetuning.infrastructure.experiment_manager import ExperimentManager

RUNS_ARGUMENT = typer.Argument(None, help="Run names to compare (default: all).")
RUN_DIR_OPTION = typer.Option(..., "--run", help="Run directory to inspect.")
RUNS_DIR_OPTION = typer.Option(None, "--runs-dir", help="Runs directory (default from config).")


def _resolve_runs_dir(runs_dir: Path | None, config_dir: Path) -> Path:
    if runs_dir is not None:
        return runs_dir
    return load_cli_config(config_dir, None).experiment.runs_dir


def list_experiments(
    runs_dir: Path = RUNS_DIR_OPTION,
    config_dir: Path = CONFIG_DIR_OPTION,
) -> None:
    """List all experiment runs with their key results."""
    resolved_runs_dir = _resolve_runs_dir(runs_dir, config_dir)
    try:
        summaries = CompareExperiments(resolved_runs_dir).execute()
    except PlatformError as error:
        console.print(f"[yellow]{error}[/yellow]")
        raise typer.Exit(code=1) from error

    table = Table(title=f"Experiments in {resolved_runs_dir}")
    for column in ("Run", "Status", "Method", "Eval loss", "Train loss", "Time (s)", "VRAM (MB)"):
        table.add_column(column)
    for summary in summaries:
        table.add_row(
            summary.name,
            summary.status,
            summary.method,
            f"{summary.eval_loss:.4f}" if summary.eval_loss is not None else "-",
            f"{summary.train_loss:.4f}" if summary.train_loss is not None else "-",
            f"{summary.total_seconds:.0f}" if summary.total_seconds is not None else "-",
            f"{summary.peak_vram_mb:.0f}" if summary.peak_vram_mb is not None else "-",
        )
    console.print(table)


def show_experiment(run_dir: Path = RUN_DIR_OPTION) -> None:
    """Show the full manifest and evaluation summary of one run."""
    try:
        manifest = ExperimentManager.load_manifest(run_dir)
        summary = summarize_run(run_dir)
    except PlatformError as error:
        console.print(f"[red]Error:[/red] {error}")
        raise typer.Exit(code=1) from error

    table = Table(title=f"Experiment {summary.name}")
    table.add_column("Property")
    table.add_column("Value")
    table.add_row("Experiment ID", manifest["experiment_id"])
    table.add_row("Status", summary.status)
    table.add_row("Model", f"{summary.model} ({manifest['model_revision']})")
    table.add_row("Method", summary.method)
    table.add_row("Seed", str(manifest["seed"]))
    table.add_row("Git commit", manifest.get("git_commit") or "-")
    table.add_row("Config hash", manifest["config_hash"][:16] + "…")
    table.add_row("Dataset", manifest["dataset"]["name"])
    table.add_row("Dataset hash", manifest["dataset"]["source_sha256"][:16] + "…")
    table.add_row(
        "Records / tokens",
        f"{manifest['dataset']['num_records']} / {manifest['dataset']['num_tokens']}",
    )
    table.add_row("Learning rate", f"{summary.learning_rate:.6g}")
    table.add_row("Optimizer / scheduler", f"{summary.optimizer} / {summary.scheduler}")
    table.add_row(
        "LoRA r/alpha/dropout", f"{summary.lora_r}/{summary.lora_alpha}/{summary.lora_dropout}"
    )
    table.add_row("Effective batch size", str(summary.effective_batch_size))
    table.add_row("Train loss", f"{summary.train_loss:.4f}" if summary.train_loss else "-")
    table.add_row("Eval loss", f"{summary.eval_loss:.4f}" if summary.eval_loss else "-")
    table.add_row("Total time (s)", str(summary.total_seconds or "-"))
    table.add_row("Peak VRAM (MB)", str(summary.peak_vram_mb or "-"))
    evaluation = load_evaluation_report(run_dir)
    if evaluation is not None and evaluation.get("trained"):
        trained = evaluation["trained"]
        if trained.get("perplexity") is not None:
            table.add_row("Perplexity (trained)", f"{trained['perplexity']:.4f}")
        if trained.get("benchmark"):
            table.add_row("Tokens/s (trained)", str(trained["benchmark"]["tokens_per_second"]))
    console.print(table)


def compare(
    runs: list[str] = RUNS_ARGUMENT,
    runs_dir: Path = RUNS_DIR_OPTION,
    config_dir: Path = CONFIG_DIR_OPTION,
) -> None:
    """Compare runs side by side and write a markdown report."""
    resolved_runs_dir = _resolve_runs_dir(runs_dir, config_dir)
    comparator = CompareExperiments(resolved_runs_dir)
    try:
        summaries = comparator.execute(runs or None)
    except PlatformError as error:
        console.print(f"[red]Error:[/red] {error}")
        raise typer.Exit(code=1) from error

    table = Table(title="Experiment comparison")
    for column in (
        "Run",
        "Method",
        "LR",
        "Optimizer",
        "Scheduler",
        "r/alpha/drop",
        "Batch",
        "Eval loss",
        "Time (s)",
        "VRAM (MB)",
    ):
        table.add_column(column)
    for summary in summaries:
        table.add_row(
            summary.name,
            summary.method,
            f"{summary.learning_rate:.6g}",
            summary.optimizer,
            summary.scheduler,
            f"{summary.lora_r}/{summary.lora_alpha}/{summary.lora_dropout}",
            str(summary.effective_batch_size),
            f"{summary.eval_loss:.4f}" if summary.eval_loss is not None else "-",
            f"{summary.total_seconds:.0f}" if summary.total_seconds is not None else "-",
            f"{summary.peak_vram_mb:.0f}" if summary.peak_vram_mb is not None else "-",
        )
    console.print(table)
    report_path = comparator.write_report(summaries)
    console.print(f"Report: [green]{report_path}[/green]")
