"""Perplexity computation over a text dataset."""

import math
from typing import Any

import torch
from loguru import logger

from finetuning.core.exceptions import EvaluationError


def compute_perplexity(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    context_length: int,
) -> float:
    """Compute corpus perplexity as exp of the token-weighted mean NLL."""
    if not texts:
        raise EvaluationError("cannot compute perplexity on an empty text list")
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=context_length
            )
            input_ids = encoded.input_ids.to(model.device)
            if input_ids.size(1) < 2:
                continue
            output = model(input_ids=input_ids, labels=input_ids)
            predicted_tokens = input_ids.size(1) - 1
            total_nll += float(output.loss) * predicted_tokens
            total_tokens += predicted_tokens
    if total_tokens == 0:
        raise EvaluationError("no usable sequences for perplexity computation")
    perplexity = math.exp(total_nll / total_tokens)
    logger.info(
        "Perplexity over {} texts ({} tokens): {:.4f}", len(texts), total_tokens, perplexity
    )
    return perplexity
