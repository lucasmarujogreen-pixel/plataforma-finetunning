"""Use case: summarize, compare and report on experiment runs."""

import json
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

from finetuning.core.exceptions import ExperimentError
from finetuning.infrastructure.experiment_manager import (
    MANIFEST_FILENAME,
    ExperimentManager,
    load_run_config,
)
from finetuning.training.callbacks import METRICS_JSONL_FILENAME


@dataclass(frozen=True)
class ExperimentSummary:
    name: str
    status: str
    model: str
    method: str
    learning_rate: float
    optimizer: str
    scheduler: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    effective_batch_size: int
    train_loss: float | None
    eval_loss: float | None
    total_seconds: float | None
    peak_vram_mb: float | None


def summarize_run(run_dir: Path) -> ExperimentSummary:
    manifest = ExperimentManager.load_manifest(run_dir)
    config = load_run_config(run_dir)
    final_metrics: dict[str, Any] = manifest.get("final_metrics") or {}
    return ExperimentSummary(
        name=manifest["name"],
        status=manifest["status"],
        model=manifest["model_name"],
        method=manifest["training_method"],
        learning_rate=config.optimizer.learning_rate,
        optimizer=config.optimizer.type.value,
        scheduler=config.scheduler.type.value,
        lora_r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        effective_batch_size=(
            config.training.micro_batch_size * config.training.gradient_accumulation_steps
        ),
        train_loss=final_metrics.get("train_loss"),
        eval_loss=final_metrics.get("eval_loss"),
        total_seconds=manifest.get("total_seconds"),
        peak_vram_mb=_peak_vram(run_dir),
    )


def _peak_vram(run_dir: Path) -> float | None:
    metrics_path = run_dir / "metrics" / METRICS_JSONL_FILENAME
    if not metrics_path.exists():
        return None
    frame = pl.read_ndjson(metrics_path)
    if "sys_vram_used_mb" not in frame.columns:
        return None
    peak = frame["sys_vram_used_mb"].cast(pl.Float64).max()
    return round(float(peak), 1) if isinstance(peak, int | float) else None


class CompareExperiments:
    def __init__(self, runs_dir: Path) -> None:
        self._runs_dir = runs_dir

    def list_run_directories(self) -> list[Path]:
        if not self._runs_dir.is_dir():
            return []
        return sorted(
            path for path in self._runs_dir.iterdir() if (path / MANIFEST_FILENAME).is_file()
        )

    def execute(self, run_names: list[str] | None = None) -> list[ExperimentSummary]:
        directories = self.list_run_directories()
        if run_names:
            wanted = set(run_names)
            directories = [path for path in directories if path.name in wanted]
            missing = wanted - {path.name for path in directories}
            if missing:
                raise ExperimentError(f"runs not found: {', '.join(sorted(missing))}")
        if not directories:
            raise ExperimentError(f"no experiment runs found in {self._runs_dir}")
        return [summarize_run(path) for path in directories]

    def write_report(self, summaries: list[ExperimentSummary]) -> Path:
        reports_dir = self._runs_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
        report_path = reports_dir / f"comparison_{timestamp}.md"

        column_names = [field.name for field in fields(ExperimentSummary)]
        header = "| " + " | ".join(column_names) + " |"
        separator = "| " + " | ".join("---" for _ in column_names) + " |"
        rows = [
            "| " + " | ".join(_format_cell(getattr(summary, name)) for name in column_names) + " |"
            for summary in summaries
        ]
        best = _best_run(summaries)
        lines = [
            "# Experiment comparison",
            "",
            f"Generated at {datetime.now(UTC).isoformat()}",
            "",
            header,
            separator,
            *rows,
            "",
        ]
        if best is not None:
            lines.append(f"Best eval_loss: **{best.name}** ({best.eval_loss})")
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("Comparison report written to {}", report_path)
        return report_path


def _format_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _best_run(summaries: list[ExperimentSummary]) -> ExperimentSummary | None:
    with_eval = [summary for summary in summaries if summary.eval_loss is not None]
    if not with_eval:
        return None
    return min(with_eval, key=lambda summary: summary.eval_loss or float("inf"))


def load_evaluation_report(run_dir: Path) -> dict[str, Any] | None:
    report_path = run_dir / "evaluation" / "evaluation.json"
    if not report_path.is_file():
        return None
    return json.loads(report_path.read_text(encoding="utf-8"))
