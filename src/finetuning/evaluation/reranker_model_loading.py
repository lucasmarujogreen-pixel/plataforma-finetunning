"""Model loading helpers for reranker evaluation."""

from pathlib import Path
from typing import Any

from sentence_transformers.cross_encoder import CrossEncoder

from finetuning.core.config.reranker_schemas import RerankerAppConfig
from finetuning.core.enums import AttentionImplementation, DeviceType, Precision
from finetuning.core.exceptions import EvaluationError
from finetuning.training.quantization import to_torch_dtype


def load_trained_reranker(
    config: RerankerAppConfig,
    model_dir: Path,
    precision: Precision,
    attention: AttentionImplementation,
    device: DeviceType,
) -> CrossEncoder:
    """Load a trained reranker from a run's model directory.

    Training always merges LoRA into the base weights before the final save
    (``training/reranker_strategies.py::merge_lora_for_saving``), so every
    saved run directory is a plain, self-contained CrossEncoder checkpoint —
    no adapter reconstruction needed here, unlike the causal-LM
    ``load_trained_model`` this mirrors.
    """
    if not (model_dir / "config.json").is_file():
        raise EvaluationError(f"no trained reranker found in {model_dir}")
    model_kwargs: dict[str, Any] = {"dtype": to_torch_dtype(precision)}
    if attention is not AttentionImplementation.AUTO:
        model_kwargs["attn_implementation"] = attention.value
    return CrossEncoder(
        str(model_dir),
        device=device.value,
        model_kwargs=model_kwargs,
        num_labels=1,
        max_length=config.training.max_length,
    )
