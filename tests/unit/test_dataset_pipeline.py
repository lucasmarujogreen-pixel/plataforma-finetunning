import json
from pathlib import Path

import polars as pl
import pytest

from finetuning.core.config.schemas import DatasetConfig
from finetuning.core.enums import DatasetFileFormat, DatasetSchema
from finetuning.core.exceptions import DatasetError
from finetuning.preprocessing.loaders import load_raw_dataset
from finetuning.preprocessing.pipeline import DatasetPipeline

PT_RECORDS = [
    {
        "messages": [
            {"role": "user", "content": f"O que é um modelo de linguagem número {index}?"},
            {"role": "assistant", "content": "É um modelo que prevê o próximo token de um texto."},
        ]
    }
    for index in range(20)
]


class FakeTokenizer:
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        return "\n".join(f"<{m['role']}>{m['content']}" for m in conversation) + "\n"

    def encode(self, text: str) -> list[int]:
        return [len(word) for word in text.split()]


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_config(tmp_path: Path, source: Path, **kwargs: object) -> DatasetConfig:
    defaults: dict[str, object] = {
        "name": "test",
        "path": str(source),
        "format": DatasetFileFormat.JSONL,
        "record_schema": DatasetSchema.CHAT,
        "cache_dir": tmp_path / "cache",
        "processed_dir": tmp_path / "processed",
    }
    defaults.update(kwargs)
    return DatasetConfig(**defaults)  # type: ignore[arg-type]


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    source = tmp_path / "data.jsonl"
    write_jsonl(source, PT_RECORDS)
    config = make_config(tmp_path, source)

    prepared = DatasetPipeline(config, FakeTokenizer(), seed=42).run()

    assert prepared.metadata.num_records == 20
    assert prepared.metadata.num_tokens > 0
    assert prepared.metadata.max_tokens >= prepared.metadata.mean_tokens
    assert prepared.metadata.language == "pt"
    assert set(prepared.dataset.keys()) == {"train", "validation"}
    assert prepared.output_dir is not None
    assert (prepared.output_dir / "metadata.json").exists()
    assert "text" in prepared.dataset["train"].column_names


def test_pipeline_removes_duplicates(tmp_path: Path) -> None:
    source = tmp_path / "data.jsonl"
    write_jsonl(source, PT_RECORDS + PT_RECORDS[:5])
    config = make_config(tmp_path, source)

    prepared = DatasetPipeline(config, FakeTokenizer(), seed=42).run(save=False)

    assert prepared.metadata.duplicates_removed == 5
    assert prepared.metadata.num_records == 20


def test_pipeline_cache_hit(tmp_path: Path) -> None:
    source = tmp_path / "data.jsonl"
    write_jsonl(source, PT_RECORDS)
    config = make_config(tmp_path, source)
    pipeline = DatasetPipeline(config, FakeTokenizer(), seed=42)

    first = pipeline.run()
    second = pipeline.run()

    assert second.metadata.created_at == first.metadata.created_at


def test_pipeline_rejects_invalid_records(tmp_path: Path) -> None:
    source = tmp_path / "data.jsonl"
    write_jsonl(source, [{"messages": [{"role": "robot", "content": "hi"}]}])
    config = make_config(tmp_path, source)

    with pytest.raises(DatasetError, match="validation failed"):
        DatasetPipeline(config, FakeTokenizer(), seed=42).run(save=False)


def test_pipeline_rejects_streaming(tmp_path: Path) -> None:
    source = tmp_path / "data.jsonl"
    write_jsonl(source, PT_RECORDS)
    config = make_config(tmp_path, source, streaming=True)

    with pytest.raises(DatasetError, match="streaming"):
        DatasetPipeline(config, FakeTokenizer(), seed=42).run()


def test_cleaning_drops_short_text_records(tmp_path: Path) -> None:
    source = tmp_path / "data.jsonl"
    records = [{"text": "conteúdo suficientemente longo para permanecer"}, {"text": "curto"}]
    write_jsonl(source, records)
    config = make_config(
        tmp_path,
        source,
        record_schema=DatasetSchema.TEXT,
        cleaning={"min_chars": 10},
        split={"validation": 0.0, "test": 0.0},
        deduplicate=False,
    )

    prepared = DatasetPipeline(config, FakeTokenizer(), seed=42).run(save=False)

    assert prepared.metadata.records_dropped_by_cleaning == 1
    assert prepared.metadata.num_records == 1


def test_load_raw_dataset_csv(tmp_path: Path) -> None:
    source = tmp_path / "data.csv"
    pl.DataFrame({"text": ["primeira linha de texto", "segunda linha de texto"]}).write_csv(source)
    config = make_config(
        tmp_path, source, format=DatasetFileFormat.CSV, record_schema=DatasetSchema.TEXT
    )

    dataset = load_raw_dataset(config)

    assert len(dataset) == 2


def test_load_raw_dataset_parquet(tmp_path: Path) -> None:
    source = tmp_path / "data.parquet"
    pl.DataFrame({"text": ["primeira linha de texto", "segunda linha de texto"]}).write_parquet(
        source
    )
    config = make_config(
        tmp_path, source, format=DatasetFileFormat.PARQUET, record_schema=DatasetSchema.TEXT
    )

    dataset = load_raw_dataset(config)

    assert len(dataset) == 2


def test_load_raw_dataset_missing_path(tmp_path: Path) -> None:
    config = make_config(tmp_path, tmp_path / "missing.jsonl")

    with pytest.raises(DatasetError, match="not found"):
        load_raw_dataset(config)
