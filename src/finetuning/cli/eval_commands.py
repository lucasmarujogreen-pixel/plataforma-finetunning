"""CLI commands for evaluation and benchmarking."""

from pathlib import Path

import typer
from rich.table import Table

from finetuning.application.evaluate_model import EvaluateModel, EvaluationReport, ModelEvaluation
from finetuning.cli.common import console
from finetuning.core.exceptions import PlatformError

RUN_DIR_OPTION = typer.Option(..., "--run", help="Run directory to evaluate.")


def _add_variant_rows(table: Table, label: str, evaluation: ModelEvaluation) -> None:
    if evaluation.perplexity is not None:
        table.add_row(f"{label} perplexity", f"{evaluation.perplexity:.4f}")
    if evaluation.benchmark is not None:
        table.add_row(f"{label} tokens/s", f"{evaluation.benchmark.tokens_per_second:.2f}")
        table.add_row(f"{label} s/token", f"{evaluation.benchmark.seconds_per_token:.6f}")
        table.add_row(
            f"{label} mean latency (s)", f"{evaluation.benchmark.mean_latency_seconds:.4f}"
        )


def _print_report(report: EvaluationReport) -> None:
    table = Table(title=f"Evaluation: {report.run_name}")
    table.add_column("Metric")
    table.add_column("Value")
    _add_variant_rows(table, "trained", report.trained)
    if report.base is not None:
        _add_variant_rows(table, "base", report.base)
    if report.perplexity_improvement is not None:
        table.add_row("perplexity improvement", f"{report.perplexity_improvement:.4f}")
    console.print(table)


def evaluate(run_dir: Path = RUN_DIR_OPTION) -> None:
    """Evaluate a trained run: perplexity, benchmark and base comparison."""
    try:
        report = EvaluateModel().execute(run_dir)
    except PlatformError as error:
        console.print(f"[red]Evaluation error:[/red] {error}")
        raise typer.Exit(code=1) from error
    _print_report(report)


def benchmark(run_dir: Path = RUN_DIR_OPTION) -> None:
    """Benchmark inference speed of a trained run."""
    try:
        report = EvaluateModel().execute(run_dir, benchmark_only=True)
    except PlatformError as error:
        console.print(f"[red]Benchmark error:[/red] {error}")
        raise typer.Exit(code=1) from error
    _print_report(report)
