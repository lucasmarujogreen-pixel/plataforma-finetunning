from pathlib import Path

import pytest

from finetuning.core.config import RerankerAppConfig, load_config
from finetuning.core.enums import OptimizerType, RerankerLossType, RerankerTrainingMethod
from finetuning.core.exceptions import ConfigurationError


def _load(config_dir: Path, overrides: list[str] | None = None) -> RerankerAppConfig:
    return load_config(
        config_dir, config_name="reranker", overrides=overrides, schema=RerankerAppConfig
    )


def test_load_default_reranker_config(config_dir: Path) -> None:
    config = _load(config_dir)

    assert isinstance(config, RerankerAppConfig)
    assert config.model.name == "BAAI/bge-reranker-v2-m3"
    assert config.training.method is RerankerTrainingMethod.LORA
    assert config.training.loss_type is RerankerLossType.LAMBDA
    assert config.optimizer.type is OptimizerType.ADAMW_TORCH
    assert config.lora.quantization.load_in_4bit is False
    assert config.evaluation.selection.top_k == 2


def test_overrides_are_applied(config_dir: Path) -> None:
    config = _load(
        config_dir,
        overrides=["training.micro_batch_size=8", "lora.r=8"],
    )

    assert config.training.micro_batch_size == 8
    assert config.lora.r == 8


def test_group_swap_via_override(config_dir: Path) -> None:
    config = _load(
        config_dir, overrides=["reranker/training@training=full_ft", "training.method=full_ft"]
    )

    assert config.training.method is RerankerTrainingMethod.FULL_FT


def test_invalid_value_rejected(config_dir: Path) -> None:
    with pytest.raises(ConfigurationError):
        _load(config_dir, overrides=["optimizer.learning_rate=-1"])


def test_unknown_key_rejected(config_dir: Path) -> None:
    with pytest.raises(ConfigurationError):
        _load(config_dir, overrides=["+training.bogus_option=1"])


def test_default_app_config_still_loads(config_dir: Path) -> None:
    """Guards against the schema-generic loader/CLI refactor breaking the causal-LM vertical."""
    from finetuning.core.config import AppConfig

    config = load_config(config_dir)

    assert isinstance(config, AppConfig)
