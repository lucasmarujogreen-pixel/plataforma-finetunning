"""Typer application entry point."""

import os

import typer

from finetuning.cli import (
    dataset_commands,
    eval_commands,
    experiment_commands,
    export_commands,
    hardware_commands,
    model_commands,
    reranker_commands,
    train_commands,
)
from finetuning.core.logging import setup_logging

app = typer.Typer(
    name="ft",
    help="Local LLM fine-tuning platform.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)

app.command("system-info")(hardware_commands.system_info)
app.command("gpu-info")(hardware_commands.gpu_info)
app.command("download-model")(model_commands.download_model)
app.command("clean-cache")(model_commands.clean_cache)
app.command("prepare-dataset")(dataset_commands.prepare_dataset)
app.command("tokenize")(dataset_commands.tokenize)
app.command("validate")(dataset_commands.validate)
app.command("train")(train_commands.train)
app.command("resume")(train_commands.resume)
app.command("evaluate")(eval_commands.evaluate)
app.command("benchmark")(eval_commands.benchmark)
app.command("list-experiments")(experiment_commands.list_experiments)
app.command("show-experiment")(experiment_commands.show_experiment)
app.command("compare")(experiment_commands.compare)
app.command("export")(export_commands.export)
app.command("merge-lora")(export_commands.merge_lora)
app.command("train-reranker")(reranker_commands.train_reranker)
app.command("evaluate-reranker")(reranker_commands.evaluate_reranker)


@app.callback()
def main(
    log_level: str = typer.Option("INFO", "--log-level", help="Console log level."),
) -> None:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    setup_logging(level=log_level)
