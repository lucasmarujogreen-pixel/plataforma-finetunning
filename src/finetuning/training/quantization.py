"""BitsAndBytes quantization configuration and dtype resolution."""

import torch
from transformers import BitsAndBytesConfig

from finetuning.core.config.schemas import QuantizationConfig
from finetuning.core.enums import Precision
from finetuning.core.exceptions import ConfigurationError

_TORCH_DTYPES = {
    Precision.BF16: torch.bfloat16,
    Precision.FP16: torch.float16,
    Precision.FP32: torch.float32,
}


def to_torch_dtype(precision: Precision) -> torch.dtype:
    dtype = _TORCH_DTYPES.get(precision)
    if dtype is None:
        raise ConfigurationError(f"precision '{precision}' must be resolved before dtype mapping")
    return dtype


def build_quantization_config(config: QuantizationConfig) -> BitsAndBytesConfig | None:
    if not config.load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.quant_type.value,
        bnb_4bit_use_double_quant=config.use_double_quant,
        bnb_4bit_compute_dtype=to_torch_dtype(config.compute_precision),
    )
