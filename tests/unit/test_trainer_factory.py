from pathlib import Path

import pytest

from finetuning.core.config import AppConfig, load_config
from finetuning.core.enums import Precision
from finetuning.infrastructure.experiment_manager import ExperimentRun
from finetuning.training.trainer_factory import build_sft_config


@pytest.fixture()
def app_config(config_dir: Path) -> AppConfig:
    return load_config(config_dir)


def make_run(tmp_path: Path) -> ExperimentRun:
    return ExperimentRun(experiment_id="id", name="test-run", root=tmp_path / "run")


def test_build_sft_config_maps_core_fields(tmp_path: Path, app_config: AppConfig) -> None:
    run = make_run(tmp_path)

    sft_config = build_sft_config(
        app_config, run, Precision.BF16, has_eval_split=True, report_to=["tensorboard"]
    )

    assert sft_config.output_dir == str(run.checkpoints_dir)
    assert sft_config.learning_rate == app_config.optimizer.learning_rate
    assert sft_config.optim == "paged_adamw_8bit"
    assert sft_config.lr_scheduler_type == "cosine"
    assert sft_config.max_length == app_config.training.context_length
    assert sft_config.bf16 is True
    assert sft_config.fp16 is False
    assert sft_config.eval_strategy == "steps"
    assert sft_config.load_best_model_at_end is True
    assert sft_config.metric_for_best_model == "eval_loss"
    assert sft_config.dataset_text_field == "text"


def test_build_sft_config_without_eval_split(tmp_path: Path, app_config: AppConfig) -> None:
    sft_config = build_sft_config(
        app_config, make_run(tmp_path), Precision.FP16, has_eval_split=False, report_to=[]
    )

    assert sft_config.eval_strategy == "no"
    assert sft_config.load_best_model_at_end is False
    assert sft_config.fp16 is True
    assert sft_config.bf16 is False


def test_build_sft_config_disables_prefetch_without_workers(
    config_dir: Path, tmp_path: Path
) -> None:
    config = load_config(config_dir, overrides=["hardware.num_workers=0"])

    sft_config = build_sft_config(
        config, make_run(tmp_path), Precision.BF16, has_eval_split=True, report_to=[]
    )

    assert sft_config.dataloader_prefetch_factor is None
    assert sft_config.dataloader_persistent_workers is False
