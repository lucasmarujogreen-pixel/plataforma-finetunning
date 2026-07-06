import pytest
import torch

from finetuning.core.config.schemas import QuantizationConfig
from finetuning.core.enums import Precision, TrainingMethod
from finetuning.core.exceptions import ConfigurationError, TrainingError
from finetuning.training.quantization import build_quantization_config, to_torch_dtype
from finetuning.training.strategies import (
    FullSFTStrategy,
    LoRAStrategy,
    QLoRAStrategy,
    get_strategy,
)


def test_to_torch_dtype_mapping() -> None:
    assert to_torch_dtype(Precision.BF16) is torch.bfloat16
    assert to_torch_dtype(Precision.FP16) is torch.float16
    assert to_torch_dtype(Precision.FP32) is torch.float32


def test_to_torch_dtype_rejects_auto() -> None:
    with pytest.raises(ConfigurationError):
        to_torch_dtype(Precision.AUTO)


def test_build_quantization_config_disabled() -> None:
    assert build_quantization_config(QuantizationConfig(load_in_4bit=False)) is None


def test_build_quantization_config_4bit() -> None:
    config = build_quantization_config(QuantizationConfig(load_in_4bit=True))

    assert config is not None
    assert config.load_in_4bit is True
    assert config.bnb_4bit_quant_type == "nf4"
    assert config.bnb_4bit_use_double_quant is True
    assert config.bnb_4bit_compute_dtype is torch.bfloat16


@pytest.mark.parametrize(
    ("method", "strategy_type"),
    [
        (TrainingMethod.LORA, LoRAStrategy),
        (TrainingMethod.QLORA, QLoRAStrategy),
        (TrainingMethod.FULL_SFT, FullSFTStrategy),
    ],
)
def test_get_strategy(method: TrainingMethod, strategy_type: type) -> None:
    assert isinstance(get_strategy(method), strategy_type)


def test_get_strategy_unknown_method() -> None:
    with pytest.raises(TrainingError):
        get_strategy("diffusion")  # type: ignore[arg-type]
