import json
from pathlib import Path

import pytest

from finetuning.core.config.reranker_schemas import RerankerDatasetConfig
from finetuning.core.enums import DatasetSchema
from finetuning.core.exceptions import DatasetError
from finetuning.preprocessing.pair_dataset import (
    cap_docs_per_query,
    load_pair_dataset,
    to_training_columns,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records), encoding="utf-8"
    )


def _records(n: int) -> list[dict]:
    return [
        {
            "meta": {"analise_id": i, "norma_id": i},
            "query": f"query {i}",
            "docs": ["doc A", "doc B", "doc C"],
            "labels": [1.0, 0.0, 0.0],
            "candidate_ids": ["1", "2", "3"],
        }
        for i in range(n)
    ]


@pytest.fixture()
def dataset_config(tmp_path: Path) -> RerankerDatasetConfig:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    _write_jsonl(train_path, _records(5))
    _write_jsonl(validation_path, _records(2))
    return RerankerDatasetConfig(
        name="test-pairs",
        path=str(train_path),
        validation_path=str(validation_path),
        cache_dir=tmp_path / "cache",
    )


def test_load_pair_dataset_returns_train_and_validation_splits(
    dataset_config: RerankerDatasetConfig,
) -> None:
    prepared = load_pair_dataset(dataset_config)

    assert set(prepared.dataset.keys()) == {"train", "validation"}
    assert len(prepared.dataset["train"]) == 5
    assert len(prepared.dataset["validation"]) == 2


def test_load_pair_dataset_computes_metadata(dataset_config: RerankerDatasetConfig) -> None:
    prepared = load_pair_dataset(dataset_config)

    assert prepared.metadata.record_schema is DatasetSchema.PAIR
    assert prepared.metadata.num_records == 5
    assert prepared.metadata.mean_tokens == pytest.approx(3.0)
    assert prepared.metadata.max_tokens == 3
    assert prepared.metadata.splits == {"train": 5, "validation": 2}
    assert prepared.metadata.source_sha256 != ""


def test_load_pair_dataset_rejects_missing_query_column(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    _write_jsonl(train_path, [{"docs": ["a"], "labels": [1.0]}])
    _write_jsonl(validation_path, [{"docs": ["a"], "labels": [1.0]}])
    config = RerankerDatasetConfig(
        name="broken", path=str(train_path), validation_path=str(validation_path)
    )

    with pytest.raises(DatasetError, match="query"):
        load_pair_dataset(config)


def test_to_training_columns_drops_meta_and_candidate_ids(
    dataset_config: RerankerDatasetConfig,
) -> None:
    prepared = load_pair_dataset(dataset_config)

    trimmed = to_training_columns(prepared.dataset["train"], dataset_config)

    assert set(trimmed.column_names) == {"query", "docs", "labels"}


def test_cap_docs_per_query_leaves_short_lists_untouched(
    dataset_config: RerankerDatasetConfig,
) -> None:
    prepared = load_pair_dataset(dataset_config)

    capped = cap_docs_per_query(prepared.dataset["train"], dataset_config, max_docs=3, seed=42)

    assert capped[0]["docs"] == ["doc A", "doc B", "doc C"]


def test_cap_docs_per_query_trims_negatives_but_keeps_every_positive(
    tmp_path: Path,
) -> None:
    record = {
        "meta": {"norma_id": 1},
        "query": "q",
        "docs": ["gold", "neg1", "neg2", "neg3", "neg4"],
        "labels": [1.0, 0.0, 0.0, 0.0, 0.0],
        "candidate_ids": ["1", "2", "3", "4", "5"],
    }
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    _write_jsonl(train_path, [record])
    _write_jsonl(validation_path, [record])
    config = RerankerDatasetConfig(
        name="cap-test", path=str(train_path), validation_path=str(validation_path)
    )
    prepared = load_pair_dataset(config)

    capped = cap_docs_per_query(prepared.dataset["train"], config, max_docs=2, seed=42)

    assert len(capped[0]["docs"]) == 2
    assert "gold" in capped[0]["docs"]
    assert capped[0]["labels"].count(1.0) == 1


def test_cap_docs_per_query_is_deterministic(tmp_path: Path) -> None:
    record = {
        "meta": {"norma_id": 1},
        "query": "q",
        "docs": [f"doc{i}" for i in range(10)],
        "labels": [1.0] + [0.0] * 9,
        "candidate_ids": [str(i) for i in range(10)],
    }
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    _write_jsonl(train_path, [record])
    _write_jsonl(validation_path, [record])
    config = RerankerDatasetConfig(
        name="cap-test", path=str(train_path), validation_path=str(validation_path)
    )
    prepared = load_pair_dataset(config)

    first = cap_docs_per_query(prepared.dataset["train"], config, max_docs=4, seed=7)
    second = cap_docs_per_query(prepared.dataset["train"], config, max_docs=4, seed=7)

    assert first[0]["docs"] == second[0]["docs"]
