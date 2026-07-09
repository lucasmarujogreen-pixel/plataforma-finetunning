from pathlib import Path

import pytest

from finetuning.core.config import RerankerAppConfig, load_config
from finetuning.core.enums import Precision, RerankerLossType
from finetuning.core.exceptions import TrainingError
from finetuning.infrastructure.experiment_manager import ExperimentRun
from finetuning.training.reranker_trainer_factory import (
    build_reranker_loss,
    build_reranker_training_args,
)


class _FakeCrossEncoder:
    """Stand-in for CrossEncoder: losses only read ``.num_labels`` at construction time."""

    num_labels = 1


@pytest.fixture()
def reranker_config(config_dir: Path) -> RerankerAppConfig:
    return load_config(config_dir, config_name="reranker", schema=RerankerAppConfig)


def make_run(tmp_path: Path) -> ExperimentRun:
    return ExperimentRun(experiment_id="id", name="test-run", root=tmp_path / "run")


def test_build_reranker_training_args_maps_core_fields(
    tmp_path: Path, reranker_config: RerankerAppConfig
) -> None:
    run = make_run(tmp_path)

    args = build_reranker_training_args(
        reranker_config, run, Precision.BF16, has_eval_split=True, report_to=["tensorboard"]
    )

    assert args.output_dir == str(run.checkpoints_dir)
    assert args.learning_rate == reranker_config.optimizer.learning_rate
    assert args.optim == "adamw_torch"
    assert args.lr_scheduler_type == "cosine"
    assert args.bf16 is True
    assert args.fp16 is False
    assert args.eval_strategy == "steps"
    # Deliberately always False for the reranker — see module docstring.
    assert args.load_best_model_at_end is False
    assert args.metric_for_best_model == "eval_loss"


def test_build_reranker_training_args_without_eval_split(
    tmp_path: Path, reranker_config: RerankerAppConfig
) -> None:
    args = build_reranker_training_args(
        reranker_config, make_run(tmp_path), Precision.FP16, has_eval_split=False, report_to=[]
    )

    assert args.eval_strategy == "no"
    assert args.load_best_model_at_end is False
    assert args.fp16 is True
    assert args.bf16 is False


def test_build_reranker_loss_dispatches_lambda_loss() -> None:
    from sentence_transformers.cross_encoder.losses import LambdaLoss

    loss = build_reranker_loss(_FakeCrossEncoder(), RerankerLossType.LAMBDA, {})  # type: ignore[arg-type]

    assert isinstance(loss, LambdaLoss)


def test_build_reranker_loss_unknown_type_raises() -> None:
    with pytest.raises(TrainingError):
        build_reranker_loss(_FakeCrossEncoder(), "bogus", {})  # type: ignore[arg-type]
