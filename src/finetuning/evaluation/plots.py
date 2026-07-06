"""Evaluation and training plots."""

from pathlib import Path

import matplotlib
import polars as pl
from loguru import logger

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_loss_curves(metrics_jsonl: Path, output_path: Path) -> bool:
    """Plot train and eval loss versus step from the training metrics log."""
    if not metrics_jsonl.exists():
        logger.warning("Metrics log not found at {}, skipping loss plot", metrics_jsonl)
        return False
    frame = pl.read_ndjson(metrics_jsonl)
    figure, axis = plt.subplots(figsize=(8, 5))
    plotted = False
    for column, label in (("loss", "train loss"), ("eval_loss", "eval loss")):
        if column in frame.columns:
            series = frame.select("step", column).drop_nulls()
            if len(series):
                axis.plot(series["step"], series[column], marker="o", label=label)
                plotted = True
    if not plotted:
        plt.close(figure)
        return False
    axis.set_xlabel("step")
    axis.set_ylabel("loss")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=120)
    plt.close(figure)
    return True


def plot_metric_comparison(
    title: str, labels: list[str], values: list[float], output_path: Path
) -> None:
    """Plot a bar chart comparing one metric across model variants."""
    figure, axis = plt.subplots(figsize=(6, 4))
    bars = axis.bar(labels, values, color=["steelblue", "darkorange"][: len(labels)])
    axis.bar_label(bars, fmt="%.3f")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=120)
    plt.close(figure)
