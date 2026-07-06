"""Use case: evaluate a trained run (perplexity, benchmark, base comparison, plots)."""

import gc
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
from loguru import logger

from finetuning.application.prepare_dataset import PrepareDataset
from finetuning.core.config.schemas import AppConfig
from finetuning.core.exceptions import EvaluationError
from finetuning.evaluation.benchmark import BenchmarkResult, run_generation_benchmark
from finetuning.evaluation.model_loading import load_trained_model
from finetuning.evaluation.perplexity import compute_perplexity
from finetuning.evaluation.plots import plot_loss_curves, plot_metric_comparison
from finetuning.infrastructure.experiment_manager import load_run_config
from finetuning.monitoring.hardware import (
    detect_hardware,
    resolve_attention,
    resolve_device,
    resolve_precision,
)
from finetuning.preprocessing.stages import TokenizerLike
from finetuning.tokenization.loader import load_tokenizer
from finetuning.training.callbacks import METRICS_JSONL_FILENAME
from finetuning.training.strategies import load_base_model

EVALUATION_REPORT_FILENAME = "evaluation.json"
_PROMPT_MAX_CHARS = 200


@dataclass(frozen=True)
class ModelEvaluation:
    perplexity: float | None
    benchmark: BenchmarkResult | None


@dataclass(frozen=True)
class EvaluationReport:
    run_name: str
    created_at: str
    trained: ModelEvaluation
    base: ModelEvaluation | None
    perplexity_improvement: float | None


class EvaluateModel:
    def execute(self, run_dir: Path, benchmark_only: bool = False) -> EvaluationReport:
        config = load_run_config(run_dir)
        profile = detect_hardware()
        device = resolve_device(profile, config.hardware)
        precision = resolve_precision(profile, config.model.precision)
        attention = resolve_attention(profile, config.model.attention)

        tokenizer = load_tokenizer(config.model, config.tokenizer)
        prepared = PrepareDataset(cast(TokenizerLike, tokenizer)).execute(config)
        split = prepared.dataset.get("validation") or prepared.dataset["train"]
        texts: list[str] = list(split["text"])
        if config.evaluation.max_eval_samples is not None:
            texts = texts[: config.evaluation.max_eval_samples]
        prompts = [
            text[:_PROMPT_MAX_CHARS] for text in texts[: config.evaluation.benchmark.num_prompts]
        ]

        model_dir = run_dir / "model"
        if not model_dir.is_dir():
            raise EvaluationError(f"model directory not found in {run_dir}")

        compute_ppl = config.evaluation.perplexity and not benchmark_only
        logger.info("Evaluating trained model from {}", model_dir)
        trained_model = load_trained_model(config, model_dir, precision, attention, device)
        trained = self._evaluate_variant(
            trained_model, tokenizer, texts, prompts, config, compute_ppl
        )
        self._release(trained_model)

        base: ModelEvaluation | None = None
        if config.evaluation.compare_with_base and not benchmark_only:
            logger.info("Evaluating base model {} for comparison", config.model.name)
            base_model = load_base_model(config, precision, attention, device)
            base = self._evaluate_variant(
                base_model, tokenizer, texts, prompts, config, compute_ppl
            )
            self._release(base_model)

        perplexity_improvement = None
        if base is not None and base.perplexity is not None and trained.perplexity is not None:
            perplexity_improvement = round(base.perplexity - trained.perplexity, 4)

        report = EvaluationReport(
            run_name=run_dir.name,
            created_at=datetime.now(UTC).isoformat(),
            trained=trained,
            base=base,
            perplexity_improvement=perplexity_improvement,
        )
        self._persist(report, run_dir, config)
        return report

    @staticmethod
    def _evaluate_variant(
        model: Any,
        tokenizer: Any,
        texts: list[str],
        prompts: list[str],
        config: AppConfig,
        compute_ppl: bool,
    ) -> ModelEvaluation:
        perplexity = (
            compute_perplexity(model, tokenizer, texts, config.training.context_length)
            if compute_ppl
            else None
        )
        benchmark = run_generation_benchmark(
            model, tokenizer, prompts, config.evaluation.benchmark.max_new_tokens
        )
        return ModelEvaluation(perplexity=perplexity, benchmark=benchmark)

    @staticmethod
    def _release(model: Any) -> None:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _persist(report: EvaluationReport, run_dir: Path, config: AppConfig) -> None:
        evaluation_dir = run_dir / "evaluation"
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        (evaluation_dir / EVALUATION_REPORT_FILENAME).write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if not config.evaluation.save_plots:
            return
        plots_dir = run_dir / "plots"
        plot_loss_curves(run_dir / "metrics" / METRICS_JSONL_FILENAME, plots_dir / "loss.png")
        if report.base is not None and report.base.perplexity is not None:
            plot_metric_comparison(
                "Perplexity (lower is better)",
                ["base", "trained"],
                [report.base.perplexity, report.trained.perplexity or 0.0],
                plots_dir / "perplexity.png",
            )
        if report.base is not None and report.base.benchmark is not None:
            plot_metric_comparison(
                "Tokens per second",
                ["base", "trained"],
                [
                    report.base.benchmark.tokens_per_second,
                    report.trained.benchmark.tokens_per_second if report.trained.benchmark else 0.0,
                ],
                plots_dir / "tokens_per_second.png",
            )
        logger.info("Evaluation artifacts saved to {} and {}", evaluation_dir, plots_dir)
