"""Independent dataset pipeline stages.

Each stage takes a dataset plus the pipeline context and returns the
transformed dataset, accumulating statistics in ``context.stats``.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from datasets import Dataset, DatasetDict
from loguru import logger

from finetuning.core.config.schemas import DatasetConfig
from finetuning.core.exceptions import DatasetError
from finetuning.core.hashing import sha256_text
from finetuning.preprocessing.schema_adapters import SchemaAdapter

_MAX_REPORTED_PROBLEMS = 20
_LANGUAGE_SAMPLE_SIZE = 100

_PT_MARKERS = {"de", "que", "não", "para", "com", "uma", "os", "as", "dos", "é", "um", "em"}
_EN_MARKERS = {"the", "and", "of", "to", "is", "in", "that", "for", "with", "it", "on", "are"}


class TokenizerLike(Protocol):
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str: ...

    def encode(self, text: str) -> list[int]: ...


@dataclass
class PipelineContext:
    config: DatasetConfig
    adapter: SchemaAdapter
    tokenizer: TokenizerLike
    seed: int
    stats: dict[str, Any] = field(default_factory=dict)


def _record_text(record: dict[str, Any]) -> str:
    if record.get("messages"):
        return "\n".join(message["content"] for message in record["messages"])
    return str(record.get("text", ""))


def validate_records(dataset: Dataset, context: PipelineContext) -> Dataset:
    problems: list[str] = []
    for index, record in enumerate(dataset):
        problem = context.adapter.validate(record)
        if problem is not None:
            problems.append(f"record {index}: {problem}")
            if len(problems) >= _MAX_REPORTED_PROBLEMS:
                break
    if problems:
        raise DatasetError("dataset validation failed:\n" + "\n".join(problems))
    context.stats["raw_records"] = len(dataset)
    return dataset


def normalize_records(dataset: Dataset, context: PipelineContext) -> Dataset:
    return dataset.map(
        context.adapter.normalize,
        remove_columns=dataset.column_names,
        desc="normalize",
    )


def clean_records(dataset: Dataset, context: PipelineContext) -> Dataset:
    cleaning = context.config.cleaning

    def _clean(record: dict[str, Any]) -> dict[str, Any]:
        if not cleaning.strip_whitespace:
            return record
        if record.get("messages"):
            return {
                "messages": [
                    {"role": message["role"], "content": message["content"].strip()}
                    for message in record["messages"]
                ]
            }
        return {"text": record["text"].strip()}

    def _keep(record: dict[str, Any]) -> bool:
        length = len(_record_text(record))
        if cleaning.drop_empty and length == 0:
            return False
        return cleaning.min_chars <= length <= cleaning.max_chars

    cleaned = dataset.map(_clean, desc="clean")
    kept = cleaned.filter(_keep, desc="length-filter")
    context.stats["records_dropped_by_cleaning"] = len(dataset) - len(kept)
    return kept


def deduplicate_records(dataset: Dataset, context: PipelineContext) -> Dataset:
    if not context.config.deduplicate:
        context.stats["duplicates_removed"] = 0
        return dataset
    seen: set[str] = set()

    def _is_first_occurrence(record: dict[str, Any]) -> bool:
        key = sha256_text(json.dumps(record, sort_keys=True, ensure_ascii=False))
        if key in seen:
            return False
        seen.add(key)
        return True

    deduplicated = dataset.filter(_is_first_occurrence, desc="deduplicate")
    context.stats["duplicates_removed"] = len(dataset) - len(deduplicated)
    return deduplicated


def detect_language(dataset: Dataset, context: PipelineContext) -> Dataset:
    sample_size = min(_LANGUAGE_SAMPLE_SIZE, len(dataset))
    words: list[str] = []
    for record in dataset.select(range(sample_size)):
        words.extend(_record_text(record).lower().split())
    vocabulary = set(words)
    pt_hits = len(vocabulary & _PT_MARKERS)
    en_hits = len(vocabulary & _EN_MARKERS)
    if pt_hits == en_hits == 0:
        context.stats["language"] = "unknown"
    else:
        context.stats["language"] = "pt" if pt_hits >= en_hits else "en"
    return dataset


def render_and_count_tokens(dataset: Dataset, context: PipelineContext) -> Dataset:
    tokenizer = context.tokenizer

    def _render(record: dict[str, Any]) -> dict[str, Any]:
        if record.get("messages"):
            text = tokenizer.apply_chat_template(
                record["messages"], tokenize=False, add_generation_prompt=False
            )
        else:
            text = record["text"]
        return {"text": text, "num_tokens": len(tokenizer.encode(text))}

    rendered = dataset.map(
        _render,
        remove_columns=dataset.column_names,
        desc="tokenize",
    )
    token_counts = rendered["num_tokens"]
    context.stats["num_tokens"] = int(sum(token_counts))
    context.stats["mean_tokens"] = float(sum(token_counts) / len(token_counts))
    context.stats["max_tokens"] = int(max(token_counts))
    return rendered


def split_records(dataset: Dataset, context: PipelineContext) -> DatasetDict:
    split = context.config.split
    holdout_fraction = split.validation + split.test
    if holdout_fraction == 0:
        result = DatasetDict({"train": dataset})
    else:
        first = dataset.train_test_split(
            test_size=holdout_fraction,
            shuffle=split.shuffle,
            seed=context.seed,
        )
        if split.test == 0:
            result = DatasetDict({"train": first["train"], "validation": first["test"]})
        elif split.validation == 0:
            result = DatasetDict({"train": first["train"], "test": first["test"]})
        else:
            second = first["test"].train_test_split(
                test_size=split.test / holdout_fraction,
                shuffle=split.shuffle,
                seed=context.seed,
            )
            result = DatasetDict(
                {
                    "train": first["train"],
                    "validation": second["train"],
                    "test": second["test"],
                }
            )
    context.stats["splits"] = {name: len(part) for name, part in result.items()}
    logger.info("Dataset splits: {}", context.stats["splits"])
    return result
