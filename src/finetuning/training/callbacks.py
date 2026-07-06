"""Trainer callbacks for metrics recording and throughput estimation."""

import json
import time
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from finetuning.monitoring.system_metrics import SystemMetricsCollector

METRICS_JSONL_FILENAME = "training_log.jsonl"
METRICS_CSV_FILENAME = "training_log.csv"


class MetricsRecorderCallback(TrainerCallback):
    """Streams training logs plus system metrics to JSONL, CSV and MLflow."""

    def __init__(
        self,
        metrics_dir: Path,
        tokens_per_step: int,
        collector: SystemMetricsCollector | None = None,
        mlflow_enabled: bool = False,
    ) -> None:
        self._metrics_dir = metrics_dir
        self._tokens_per_step = tokens_per_step
        self._collector = collector
        self._mlflow_enabled = mlflow_enabled
        self._jsonl_path = metrics_dir / METRICS_JSONL_FILENAME
        self._last_log_time: float | None = None
        self._last_log_step: int = 0

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if logs is None or not state.is_world_process_zero:
            return
        row: dict[str, Any] = {
            "timestamp": time.time(),
            "step": state.global_step,
            "epoch": round(state.epoch or 0.0, 4),
            **logs,
        }
        row.update(self._throughput(state.global_step))
        if self._collector is not None:
            row.update(self._collector.snapshot())
        self._append_jsonl(row)
        self._log_to_mlflow(row, state.global_step)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if not state.is_world_process_zero:
            return
        if self._collector is not None:
            self._collector.close()
        self._write_csv()

    def _throughput(self, global_step: int) -> dict[str, float]:
        now = time.perf_counter()
        if self._last_log_time is None or global_step <= self._last_log_step:
            self._last_log_time = now
            self._last_log_step = global_step
            return {}
        elapsed = now - self._last_log_time
        steps = global_step - self._last_log_step
        self._last_log_time = now
        self._last_log_step = global_step
        if elapsed <= 0:
            return {}
        steps_per_second = steps / elapsed
        return {
            "steps_per_second": round(steps_per_second, 4),
            "estimated_tokens_per_second": round(steps_per_second * self._tokens_per_step, 2),
        }

    def _append_jsonl(self, row: dict[str, Any]) -> None:
        self._metrics_dir.mkdir(parents=True, exist_ok=True)
        with self._jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _log_to_mlflow(self, row: dict[str, Any], step: int) -> None:
        if not self._mlflow_enabled:
            return
        try:
            import mlflow

            if mlflow.active_run() is None:
                return
            numeric = {
                key: float(value)
                for key, value in row.items()
                if isinstance(value, int | float) and key.startswith("sys_")
            }
            if numeric:
                mlflow.log_metrics(numeric, step=step)
        except Exception as error:
            logger.debug("MLflow system metrics logging failed: {}", error)

    def _write_csv(self) -> None:
        if not self._jsonl_path.exists():
            return
        try:
            pl.read_ndjson(self._jsonl_path).write_csv(self._metrics_dir / METRICS_CSV_FILENAME)
        except Exception as error:
            logger.warning("Failed to convert metrics JSONL to CSV: {}", error)
