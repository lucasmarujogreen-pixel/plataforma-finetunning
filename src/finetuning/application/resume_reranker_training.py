"""Use case: resume reranker training from the last checkpoint of a previous run."""

from pathlib import Path

from loguru import logger
from transformers.trainer_utils import get_last_checkpoint

from finetuning.application.train_reranker_model import TrainRerankerModel, TrainRerankerModelResult
from finetuning.core.exceptions import TrainingError
from finetuning.infrastructure.experiment_manager import load_reranker_run_config


class ResumeRerankerTraining:
    def execute(self, run_dir: Path) -> TrainRerankerModelResult:
        config = load_reranker_run_config(run_dir)

        checkpoints_dir = run_dir / "checkpoints"
        checkpoint = get_last_checkpoint(str(checkpoints_dir)) if checkpoints_dir.is_dir() else None
        if checkpoint is None:
            raise TrainingError(f"no checkpoint found to resume in {checkpoints_dir}")
        logger.info("Resuming reranker training from checkpoint {}", checkpoint)
        return TrainRerankerModel().execute(config, resume_from=Path(checkpoint))
