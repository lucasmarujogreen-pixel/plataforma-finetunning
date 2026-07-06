from pathlib import Path

import pytest

from finetuning.core.config import AppConfig, load_config
from finetuning.core.enums import ExportFormat, OptimizerType, TrainingMethod
from finetuning.core.exceptions import ConfigurationError


def test_load_default_config(config_dir: Path) -> None:
    config = load_config(config_dir)

    assert isinstance(config, AppConfig)
    assert config.model.name == "Qwen/Qwen3-0.6B"
    assert config.training.method is TrainingMethod.QLORA
    assert config.lora.quantization.load_in_4bit is True
    assert config.optimizer.type is OptimizerType.PAGED_ADAMW_8BIT
    assert ExportFormat.GGUF in config.export.formats


def test_overrides_are_applied(config_dir: Path) -> None:
    config = load_config(
        config_dir,
        overrides=[
            "optimizer.learning_rate=1e-4",
            "training.micro_batch_size=2",
            "lora.r=8",
        ],
    )

    assert config.optimizer.learning_rate == pytest.approx(1e-4)
    assert config.training.micro_batch_size == 2
    assert config.lora.r == 8


def test_group_swap_via_override(config_dir: Path) -> None:
    config = load_config(config_dir, overrides=["lora=lora", "training.method=lora"])

    assert config.training.method is TrainingMethod.LORA
    assert config.lora.quantization.load_in_4bit is False


def test_invalid_value_rejected(config_dir: Path) -> None:
    with pytest.raises(ConfigurationError):
        load_config(config_dir, overrides=["optimizer.learning_rate=-1"])


def test_unknown_key_rejected(config_dir: Path) -> None:
    with pytest.raises(ConfigurationError):
        load_config(config_dir, overrides=["+training.bogus_option=1"])


def test_qlora_without_4bit_rejected(config_dir: Path) -> None:
    with pytest.raises(ConfigurationError):
        load_config(config_dir, overrides=["lora=lora"])


def test_lora_with_4bit_rejected(config_dir: Path) -> None:
    with pytest.raises(ConfigurationError):
        load_config(config_dir, overrides=["training.method=lora"])


def test_warmup_ratio_and_steps_conflict(config_dir: Path) -> None:
    with pytest.raises(ConfigurationError):
        load_config(config_dir, overrides=["scheduler.warmup_steps=100"])


def test_split_fractions_must_stay_below_one(config_dir: Path) -> None:
    with pytest.raises(ConfigurationError):
        load_config(
            config_dir,
            overrides=["dataset.split.validation=0.6", "dataset.split.test=0.5"],
        )


def test_missing_config_dir_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        load_config(tmp_path / "does-not-exist")
