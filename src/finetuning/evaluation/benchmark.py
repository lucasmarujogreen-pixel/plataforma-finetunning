"""Inference throughput and latency benchmark."""

import time
from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger

from finetuning.core.exceptions import EvaluationError


@dataclass(frozen=True)
class BenchmarkResult:
    num_prompts: int
    generated_tokens: int
    total_seconds: float
    tokens_per_second: float
    seconds_per_token: float
    mean_latency_seconds: float


def run_generation_benchmark(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
) -> BenchmarkResult:
    """Measure greedy generation throughput over a list of prompts."""
    if not prompts:
        raise EvaluationError("cannot benchmark with an empty prompt list")
    model.eval()
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def _generate(prompt: str) -> int:
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = encoded.input_ids.to(model.device)
        attention_mask = encoded.attention_mask.to(model.device)
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        return int(output.shape[1] - input_ids.shape[1])

    _generate(prompts[0])

    generated_tokens = 0
    latencies: list[float] = []
    start = time.perf_counter()
    for prompt in prompts:
        prompt_start = time.perf_counter()
        generated_tokens += _generate(prompt)
        latencies.append(time.perf_counter() - prompt_start)
    total_seconds = time.perf_counter() - start

    if generated_tokens == 0:
        raise EvaluationError("benchmark generated zero tokens")
    result = BenchmarkResult(
        num_prompts=len(prompts),
        generated_tokens=generated_tokens,
        total_seconds=round(total_seconds, 4),
        tokens_per_second=round(generated_tokens / total_seconds, 2),
        seconds_per_token=round(total_seconds / generated_tokens, 6),
        mean_latency_seconds=round(sum(latencies) / len(latencies), 4),
    )
    logger.info(
        "Benchmark: {} tokens in {:.2f}s ({} tok/s)",
        result.generated_tokens,
        result.total_seconds,
        result.tokens_per_second,
    )
    return result
