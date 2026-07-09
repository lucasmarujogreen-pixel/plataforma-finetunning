"""Use case: run a fine-tuning experiment end to end."""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from loguru import logger
from transformers import EarlyStoppingCallback, set_seed
from trl import SFTTrainer

from finetuning.application.prepare_dataset import PrepareDataset
from finetuning.core.config.schemas import AppConfig
from finetuning.core.enums import DeviceType
from finetuning.core.exceptions import TrainingError
from finetuning.infrastructure.experiment_manager import ExperimentManager, ExperimentRun
from finetuning.monitoring.hardware import (
    detect_hardware,
    limit_vram_usage,
    resolve_attention,
    resolve_device,
    resolve_precision,
)
from finetuning.monitoring.system_metrics import SystemMetricsCollector
from finetuning.preprocessing.stages import TokenizerLike
from finetuning.tokenization.loader import load_tokenizer
from finetuning.training.callbacks import MetricsRecorderCallback
from finetuning.training.strategies import get_strategy
from finetuning.training.trainer_factory import build_sft_config


@dataclass(frozen=True)
class TrainModelResult:
    run_dir: Path
    experiment_id: str
    total_seconds: float
    metrics: dict[str, float]


class TrainModel:
    def execute(self, config: AppConfig, resume_from: Path | None = None) -> TrainModelResult:
        profile = detect_hardware()
        device = resolve_device(profile, config.hardware)
        precision = resolve_precision(profile, config.model.precision)
        attention = resolve_attention(profile, config.model.attention)
        if device is DeviceType.CPU:
            logger.warning("Training on CPU: debug mode only, expect very low throughput")
        limit_vram_usage(device, config.hardware)
        set_seed(config.training.seed)

        tokenizer = load_tokenizer(config.model, config.tokenizer)
        prepared = PrepareDataset(cast(TokenizerLike, tokenizer)).execute(config)
        has_eval_split = "validation" in prepared.dataset

        manager = ExperimentManager(config.experiment.runs_dir)
        run = manager.create_run(
            config,
            prepared.metadata,
            profile,
            extras={
                "resolved_device": device.value,
                "resolved_precision": precision.value,
                "resolved_attention": attention.value,
                "resumed_from": str(resume_from) if resume_from else None,
            },
        )

        report_to = self._configure_reporting(config)
        strategy = get_strategy(config.training.method)
        start = time.perf_counter()
        try:
            model = strategy.build_model(config, precision, attention, device)
            sft_config = build_sft_config(config, run, precision, has_eval_split, report_to)
            trainer = SFTTrainer(
                model=model,
                args=sft_config,
                train_dataset=prepared.dataset["train"],
                eval_dataset=prepared.dataset["validation"] if has_eval_split else None,
                processing_class=tokenizer,
                callbacks=self._build_callbacks(config, run, device, has_eval_split),
            )
            train_output = trainer.train(
                resume_from_checkpoint=str(resume_from) if resume_from else None
            )
            metrics: dict[str, float] = dict(train_output.metrics)
            if has_eval_split:
                metrics.update(trainer.evaluate())
            trainer.save_model(str(run.model_dir))
            tokenizer.save_pretrained(str(run.model_dir))
        except Exception as error:
            total_seconds = time.perf_counter() - start
            manager.finalize_run(
                run, status="failed", total_seconds=total_seconds, final_metrics=None
            )
            raise TrainingError(f"training failed for run {run.name}: {error}") from error

        total_seconds = time.perf_counter() - start
        numeric_metrics = {
            key: float(value) for key, value in metrics.items() if isinstance(value, int | float)
        }
        manager.finalize_run(
            run, status="completed", total_seconds=total_seconds, final_metrics=numeric_metrics
        )
        logger.info("Training finished in {:.1f}s, artifacts at {}", total_seconds, run.root)
        return TrainModelResult(
            run_dir=run.root,
            experiment_id=run.experiment_id,
            total_seconds=total_seconds,
            metrics=numeric_metrics,
        )

    @staticmethod
    def _configure_reporting(config: AppConfig) -> list[str]:
        report_to: list[str] = []
        if config.logging.mlflow.enabled:
            os.environ["MLFLOW_TRACKING_URI"] = config.logging.mlflow.tracking_uri
            os.environ["MLFLOW_EXPERIMENT_NAME"] = config.logging.mlflow.experiment_name
            report_to.append("mlflow")
        if config.logging.tensorboard.enabled:
            report_to.append("tensorboard")
        return report_to

    @staticmethod
    def _build_callbacks(
        config: AppConfig,
        run: ExperimentRun,
        device: DeviceType,
        has_eval_split: bool,
    ) -> list[Any]:
        training = config.training
        tokens_per_step = (
            training.micro_batch_size
            * training.gradient_accumulation_steps
            * training.context_length
        )
        callbacks: list[Any] = [
            MetricsRecorderCallback(
                metrics_dir=run.metrics_dir,
                tokens_per_step=tokens_per_step,
                collector=SystemMetricsCollector() if device is DeviceType.CUDA else None,
                mlflow_enabled=config.logging.mlflow.enabled,
            )
        ]
        if training.early_stopping.enabled and has_eval_split:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=training.early_stopping.patience,
                    early_stopping_threshold=training.early_stopping.threshold,
                )
            )
        return callbacks
