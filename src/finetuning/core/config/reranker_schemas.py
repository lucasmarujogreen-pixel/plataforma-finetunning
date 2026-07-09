"""Typed configuration schemas for the reranker training vertical.

Mirrors ``core/config/schemas.py`` but targets a cross-encoder classifier
``(query, candidate) -> relevance`` instead of a causal-LM generator. Reuses
the generic building blocks (model/optimizer/scheduler/lora/logging/
experiment/hardware) directly and only defines schemas for the pieces that
are genuinely different in shape: the training loop, the dataset, and the
evaluation.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field

from finetuning.core.config.schemas import (
    BaseConfig,
    EarlyStoppingConfig,
    ExperimentConfig,
    HardwareConfig,
    LoggingConfig,
    LoraConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
)
from finetuning.core.enums import RerankerLossType, RerankerTrainingMethod


class RerankerTrainingConfig(BaseConfig):
    method: RerankerTrainingMethod = RerankerTrainingMethod.LORA
    seed: int = 42
    num_epochs: float = Field(default=3.0, gt=0)
    max_steps: int = Field(default=-1, ge=-1)
    micro_batch_size: int = Field(default=4, gt=0)
    gradient_accumulation_steps: int = Field(default=4, gt=0)
    max_length: int = Field(default=512, gt=0)
    gradient_checkpointing: bool = True
    eval_steps: int = Field(default=100, gt=0)
    save_steps: int = Field(default=100, gt=0)
    logging_steps: int = Field(default=10, gt=0)
    save_total_limit: int = Field(default=3, gt=0)
    loss_type: RerankerLossType = RerankerLossType.LAMBDA
    loss_kwargs: dict[str, float | int | str] = Field(default_factory=dict)
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()
    max_train_docs_per_query: int | None = Field(default=None, gt=0)


class RerankerDatasetConfig(BaseConfig):
    name: str = Field(min_length=1)
    path: str = Field(min_length=1)
    validation_path: str = Field(min_length=1)
    query_field: str = "query"
    docs_field: str = "docs"
    labels_field: str = "labels"
    cache_dir: Path = Path("datasets/cache")


class SelectionConfig(BaseConfig):
    strategy: Literal["top_k", "threshold"] = "top_k"
    top_k: int = Field(default=5, gt=0)
    threshold: float = Field(default=0.5, ge=0, le=1)


class RerankerEvaluationConfig(BaseConfig):
    enabled: bool = True
    reranking_evaluator: bool = True
    ndcg_at_k: list[int] = Field(default_factory=lambda: [5, 10])
    hierarchical_f1: bool = True
    train_path: Path = Path("datasets/raw/greenlegis_condicoes_train_v6.jsonl")
    test_path: Path = Path("datasets/raw/greenlegis_condicoes_test_v6.jsonl")
    knn_candidates_per_query: int = Field(default=40, gt=0)
    limit: int = Field(default=200, gt=0)
    selection: SelectionConfig = SelectionConfig()


class RerankerAppConfig(BaseConfig):
    """Root config for the reranker vertical.

    Deliberately has no ``tokenizer``/``export`` groups (a cross-encoder
    loads its own tokenizer and does not export to GGUF/Ollama) and no
    QLoRA cross-validation like ``AppConfig`` — the reranker is small enough
    (~568M params) that 4-bit quantization is not needed; add it back if a
    larger reranker base model is ever adopted.
    """

    model: ModelConfig
    training: RerankerTrainingConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    dataset: RerankerDatasetConfig
    lora: LoraConfig
    logging: LoggingConfig
    evaluation: RerankerEvaluationConfig
    experiment: ExperimentConfig
    hardware: HardwareConfig
