"""Use case: run the dataset preparation pipeline."""

from finetuning.core.config.schemas import AppConfig
from finetuning.preprocessing.pipeline import DatasetPipeline, PreparedDataset
from finetuning.preprocessing.stages import TokenizerLike


class PrepareDataset:
    def __init__(self, tokenizer: TokenizerLike) -> None:
        self._tokenizer = tokenizer

    def execute(self, config: AppConfig, save: bool = True, force: bool = False) -> PreparedDataset:
        pipeline = DatasetPipeline(
            config=config.dataset,
            tokenizer=self._tokenizer,
            seed=config.training.seed,
        )
        return pipeline.run(save=save, force=force)
