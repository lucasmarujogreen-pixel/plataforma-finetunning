import pytest

from finetuning.core.enums import RerankerTrainingMethod
from finetuning.core.exceptions import TrainingError
from finetuning.training.reranker_strategies import (
    RerankerFullFTStrategy,
    RerankerLoRAStrategy,
    get_reranker_strategy,
)


def test_get_reranker_strategy_returns_lora() -> None:
    strategy = get_reranker_strategy(RerankerTrainingMethod.LORA)

    assert isinstance(strategy, RerankerLoRAStrategy)


def test_get_reranker_strategy_returns_full_ft() -> None:
    strategy = get_reranker_strategy(RerankerTrainingMethod.FULL_FT)

    assert isinstance(strategy, RerankerFullFTStrategy)


def test_get_reranker_strategy_unknown_method_raises() -> None:
    with pytest.raises(TrainingError):
        get_reranker_strategy("bogus")  # type: ignore[arg-type]
