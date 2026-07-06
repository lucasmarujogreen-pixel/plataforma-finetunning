import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from finetuning.core.exceptions import EvaluationError
from finetuning.evaluation.benchmark import run_generation_benchmark
from finetuning.evaluation.perplexity import compute_perplexity
from finetuning.evaluation.plots import plot_loss_curves, plot_metric_comparison


class FakeCausalModel:
    device = "cpu"

    def __init__(self, loss: float = math.log(2.0), new_tokens: int = 8) -> None:
        self._loss = loss
        self._new_tokens = new_tokens

    def eval(self) -> None:
        pass

    def __call__(self, input_ids: torch.Tensor, labels: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(loss=torch.tensor(self._loss))

    def generate(self, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
        batch, length = input_ids.shape
        return torch.ones((batch, length + self._new_tokens), dtype=torch.long)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(
        self, text: str, return_tensors: str, truncation: bool, max_length: int
    ) -> SimpleNamespace:
        tokens = min(max(len(text.split()), 1), max_length)
        return SimpleNamespace(
            input_ids=torch.ones((1, tokens), dtype=torch.long),
            attention_mask=torch.ones((1, tokens), dtype=torch.long),
        )


def test_compute_perplexity_matches_constant_loss() -> None:
    perplexity = compute_perplexity(
        FakeCausalModel(loss=math.log(2.0)),
        FakeTokenizer(),
        ["uma frase de teste com tokens", "outra frase de teste"],
        context_length=128,
    )

    assert perplexity == pytest.approx(2.0, rel=1e-4)


def test_compute_perplexity_rejects_empty_texts() -> None:
    with pytest.raises(EvaluationError):
        compute_perplexity(FakeCausalModel(), FakeTokenizer(), [], context_length=128)


def test_benchmark_counts_generated_tokens() -> None:
    result = run_generation_benchmark(
        FakeCausalModel(new_tokens=8),
        FakeTokenizer(),
        ["prompt um", "prompt dois", "prompt três"],
        max_new_tokens=8,
    )

    assert result.num_prompts == 3
    assert result.generated_tokens == 24
    assert result.tokens_per_second > 0
    assert result.seconds_per_token > 0


def test_benchmark_rejects_empty_prompts() -> None:
    with pytest.raises(EvaluationError):
        run_generation_benchmark(FakeCausalModel(), FakeTokenizer(), [], max_new_tokens=8)


def test_plot_loss_curves(tmp_path: Path) -> None:
    metrics_path = tmp_path / "training_log.jsonl"
    rows = [
        {"step": 5, "loss": 2.0},
        {"step": 10, "loss": 1.5, "eval_loss": 1.8},
        {"step": 20, "loss": 1.2, "eval_loss": 1.6},
    ]
    metrics_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    output = tmp_path / "plots" / "loss.png"

    assert plot_loss_curves(metrics_path, output) is True
    assert output.exists()


def test_plot_loss_curves_missing_file(tmp_path: Path) -> None:
    assert plot_loss_curves(tmp_path / "missing.jsonl", tmp_path / "loss.png") is False


def test_plot_metric_comparison(tmp_path: Path) -> None:
    output = tmp_path / "perplexity.png"

    plot_metric_comparison("Perplexity", ["base", "trained"], [12.5, 9.8], output)

    assert output.exists()
