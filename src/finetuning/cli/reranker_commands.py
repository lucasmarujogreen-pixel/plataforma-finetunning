"""CLI commands for the reranker vertical: train-reranker, evaluate-reranker."""

from pathlib import Path

import typer
from rich.table import Table

from finetuning.application.evaluate_reranker_model import (
    EvaluateRerankerModel,
    EvaluateRerankerModelResult,
)
from finetuning.application.train_reranker_model import TrainRerankerModel, TrainRerankerModelResult
from finetuning.cli.common import CONFIG_DIR_OPTION, OVERRIDES_ARGUMENT, console, load_cli_config
from finetuning.core.config.reranker_schemas import RerankerAppConfig
from finetuning.core.exceptions import PlatformError

RUN_DIR_OPTION = typer.Option(..., "--run", help="Run directory to evaluate.")


def _print_train_result(result: TrainRerankerModelResult) -> None:
    table = Table(title="Reranker training result")
    table.add_column("Property")
    table.add_column("Value")
    table.add_row("Run", str(result.run_dir))
    table.add_row("Experiment ID", result.experiment_id)
    table.add_row("Total time (s)", f"{result.total_seconds:.1f}")
    for key in ("train_loss", "eval_loss", "train_samples_per_second"):
        if key in result.metrics:
            table.add_row(key, f"{result.metrics[key]:.4f}")
    console.print(table)


def _print_evaluation_report(result: EvaluateRerankerModelResult) -> None:
    report = result.report
    table = Table(title="Reranker evaluation (comparável à Etapa 8 da POC)")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("item_f1", f"{report['item_f1']:.4f}")
    table.add_row("item_precision", f"{report['item_precision']:.4f}")
    table.add_row("item_recall", f"{report['item_recall']:.4f}")
    table.add_row("exact_match_rate", f"{report['exact_match_rate']:.4f}")
    table.add_row("retrieval_recall_at_k", f"{report['retrieval_recall_at_k']:.4f}")
    for level, value in report["hierarchy_f1"].items():
        table.add_row(f"hierarchy_f1[{level}]", f"{value:.4f}")
    console.print(table)
    console.print(f"Relatório completo: {result.report_path}")
    console.print(f"Amostras detalhadas: {result.samples_path}")


def train_reranker(
    overrides: list[str] = OVERRIDES_ARGUMENT,
    config_dir: Path = CONFIG_DIR_OPTION,
) -> None:
    """Run a reranker fine-tuning experiment with the composed configuration."""
    config = load_cli_config(
        config_dir, overrides, config_name="reranker", schema=RerankerAppConfig
    )
    try:
        result = TrainRerankerModel().execute(config)
    except PlatformError as error:
        console.print(f"[red]Training error:[/red] {error}")
        raise typer.Exit(code=1) from error
    _print_train_result(result)


def evaluate_reranker(run_dir: Path = RUN_DIR_OPTION) -> None:
    """Evaluate a trained reranker on the POC's canonical 200-sample test set."""
    try:
        result = EvaluateRerankerModel().execute(run_dir)
    except PlatformError as error:
        console.print(f"[red]Evaluation error:[/red] {error}")
        raise typer.Exit(code=1) from error
    _print_evaluation_report(result)
