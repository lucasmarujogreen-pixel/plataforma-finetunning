"""Loads the grouped (query, docs, labels) reranker dataset.

Distinct from ``preprocessing/pipeline.py``: that pipeline renders every
record into a single ``text`` field via chat templates for causal-LM SFT.
The reranker dataset is already in its final listwise shape (one JSONL
record per query, with ``docs``/``labels`` lists) produced by
``scripts/build_reranker_dataset.py`` — no chat template, cleaning or
tokenization stage applies here, only loading and describing.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

from finetuning.core.config.reranker_schemas import RerankerDatasetConfig
from finetuning.core.enums import DatasetSchema
from finetuning.core.exceptions import DatasetError
from finetuning.core.hashing import sha256_file
from finetuning.domain.entities import DatasetMetadata


@dataclass(frozen=True)
class PreparedPairDataset:
    dataset: DatasetDict
    metadata: DatasetMetadata


def load_pair_dataset(config: RerankerDatasetConfig) -> PreparedPairDataset:
    """Load the train/validation JSONL pair files and describe them."""
    dataset = load_dataset(
        "json",
        data_files={"train": config.path, "validation": config.validation_path},
        cache_dir=str(config.cache_dir),
    )
    if not isinstance(dataset, DatasetDict):
        raise DatasetError("expected a DatasetDict with train/validation splits")
    for split_name, expected_field in (
        ("train", config.query_field),
        ("validation", config.query_field),
    ):
        if expected_field not in dataset[split_name].column_names:
            raise DatasetError(
                f"'{expected_field}' column missing from {split_name} split of {config.path}"
            )
    metadata = _compute_metadata(dataset, config)
    return PreparedPairDataset(dataset=dataset, metadata=metadata)


def _compute_metadata(dataset: DatasetDict, config: RerankerDatasetConfig) -> DatasetMetadata:
    train = dataset["train"]
    docs_per_query = [len(docs) for docs in train[config.docs_field]]
    num_records = len(train)
    total_docs = sum(docs_per_query)
    source_path = Path(config.path)
    return DatasetMetadata(
        name=config.name,
        version="v1",
        source_path=config.path,
        source_sha256=sha256_file(source_path) if source_path.is_file() else "",
        record_schema=DatasetSchema.PAIR,
        num_records=num_records,
        num_tokens=total_docs,
        mean_tokens=round(total_docs / max(num_records, 1), 2),
        max_tokens=max(docs_per_query, default=0),
        language="pt",
        created_at=datetime.now(UTC).isoformat(),
        splits={name: len(split) for name, split in dataset.items()},
    )


def to_training_columns(dataset: Dataset, config: RerankerDatasetConfig) -> Dataset:
    """Keep only the query/docs/labels columns the trainer/loss expect."""
    keep = {config.query_field, config.docs_field, config.labels_field}
    drop = [column for column in dataset.column_names if column not in keep]
    return dataset.remove_columns(drop) if drop else dataset
