"""Use case: run a reranker fine-tuning experiment end to end.

Mirrors ``application/train_model.py`` — same "casca" (hardware detection,
experiment run creation, callbacks, seed) — swapping the causal-LM-specific
middle (``SFTTrainer``/``build_sft_config``) for the reranker's
(``CrossEncoderTrainer``/``build_reranker_training_args``/``build_reranker_loss``).
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger
from sentence_transformers.cross_encoder import CrossEncoderTrainer
from transformers import EarlyStoppingCallback, set_seed

from finetuning.core.config.reranker_schemas import RerankerAppConfig
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
from finetuning.preprocessing.pair_dataset import (
    cap_docs_per_query,
    load_pair_dataset,
    to_training_columns,
)
from finetuning.training.callbacks import MetricsRecorderCallback
from finetuning.training.reranker_strategies import get_reranker_strategy, merge_lora_for_saving
from finetuning.training.reranker_trainer_factory import (
    build_reranker_loss,
    build_reranker_training_args,
)


@dataclass(frozen=True)
class TrainRerankerModelResult:
    run_dir: Path
    experiment_id: str
    total_seconds: float
    metrics: dict[str, float]


class TrainRerankerModel:
    def execute(
        self, config: RerankerAppConfig, resume_from: Path | None = None
    ) -> TrainRerankerModelResult:
        profile = detect_hardware()
        device = resolve_device(profile, config.hardware)
        precision = resolve_precision(profile, config.model.precision)
        attention = resolve_attention(profile, config.model.attention)
        if device is DeviceType.CPU:
            logger.warning("Training on CPU: debug mode only, expect very low throughput")
        limit_vram_usage(device, config.hardware)
        set_seed(config.training.seed)

        prepared = load_pair_dataset(config.dataset)
        train_dataset = to_training_columns(prepared.dataset["train"], config.dataset)
        if config.training.max_train_docs_per_query is not None:
            train_dataset = cap_docs_per_query(
                train_dataset,
                config.dataset,
                config.training.max_train_docs_per_query,
                config.training.seed,
            )
        validation_dataset = to_training_columns(prepared.dataset["validation"], config.dataset)
        has_eval_split = len(validation_dataset) > 0

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
        strategy = get_reranker_strategy(config.training.method)
        start = time.perf_counter()
        try:
            model = strategy.build_model(config, precision, attention, device)
            loss = build_reranker_loss(
                model, config.training.loss_type, config.training.loss_kwargs
            )
            training_args = build_reranker_training_args(
                config, run, precision, has_eval_split, report_to
            )
            trainer = CrossEncoderTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset if has_eval_split else None,
                loss=loss,
                callbacks=self._build_callbacks(config, run, device, has_eval_split),
            )
            train_output = trainer.train(
                resume_from_checkpoint=str(resume_from) if resume_from else None
            )
            metrics: dict[str, float] = dict(train_output.metrics)
            if has_eval_split:
                metrics.update(trainer.evaluate())
            model = merge_lora_for_saving(model)
            model.save_pretrained(str(run.model_dir))
        except Exception as error:
            total_seconds = time.perf_counter() - start
            manager.finalize_run(
                run, status="failed", total_seconds=total_seconds, final_metrics=None
            )
            raise TrainingError(f"reranker training failed for run {run.name}: {error}") from error

        total_seconds = time.perf_counter() - start
        numeric_metrics = {
            key: float(value) for key, value in metrics.items() if isinstance(value, int | float)
        }
        manager.finalize_run(
            run, status="completed", total_seconds=total_seconds, final_metrics=numeric_metrics
        )
        logger.info(
            "Reranker training finished in {:.1f}s, artifacts at {}", total_seconds, run.root
        )
        return TrainRerankerModelResult(
            run_dir=run.root,
            experiment_id=run.experiment_id,
            total_seconds=total_seconds,
            metrics=numeric_metrics,
        )

    @staticmethod
    def _configure_reporting(config: RerankerAppConfig) -> list[str]:
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
        config: RerankerAppConfig,
        run: ExperimentRun,
        device: DeviceType,
        has_eval_split: bool,
    ) -> list[Any]:
        training = config.training
        tokens_per_step = (
            training.micro_batch_size * training.gradient_accumulation_steps * training.max_length
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
