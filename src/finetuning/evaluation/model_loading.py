"""Model loading helpers for evaluation and export."""

from pathlib import Path
from typing import Any

from peft import PeftModel
from transformers import AutoModelForCausalLM

from finetuning.core.config.schemas import AppConfig
from finetuning.core.enums import AttentionImplementation, DeviceType, Precision, TrainingMethod
from finetuning.core.exceptions import EvaluationError
from finetuning.training.quantization import build_quantization_config, to_torch_dtype
from finetuning.training.strategies import load_base_model


def load_trained_model(
    config: AppConfig,
    model_dir: Path,
    precision: Precision,
    attention: AttentionImplementation,
    device: DeviceType,
) -> Any:
    """Load a trained model directory: PEFT adapter over the base, or full weights."""
    if (model_dir / "adapter_config.json").is_file():
        quantization = None
        effective_precision = precision
        if config.training.method is TrainingMethod.QLORA:
            quantization = build_quantization_config(config.lora.quantization)
            effective_precision = config.lora.quantization.compute_precision
        base = load_base_model(config, effective_precision, attention, device, quantization)
        return PeftModel.from_pretrained(base, model_dir)
    if (model_dir / "config.json").is_file():
        kwargs: dict[str, Any] = {"dtype": to_torch_dtype(precision)}
        if attention is not AttentionImplementation.AUTO:
            kwargs["attn_implementation"] = attention.value
        if device is DeviceType.CUDA:
            kwargs["device_map"] = {"": 0}
        return AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    raise EvaluationError(f"no trained model found in {model_dir}")
