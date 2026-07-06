"""Dataset preparation pipeline: load, validate, clean, dedup, tokenize, split, cache."""

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk
from loguru import logger

from finetuning.core.config.schemas import DatasetConfig
from finetuning.core.enums import DatasetFileFormat, DatasetSchema
from finetuning.core.exceptions import DatasetError
from finetuning.core.hashing import sha256_dir, sha256_file, sha256_text
from finetuning.domain.entities import DatasetMetadata
from finetuning.preprocessing.loaders import load_raw_dataset
from finetuning.preprocessing.schema_adapters import get_schema_adapter
from finetuning.preprocessing.stages import (
    PipelineContext,
    TokenizerLike,
    clean_records,
    deduplicate_records,
    detect_language,
    normalize_records,
    render_and_count_tokens,
    split_records,
    validate_records,
)

METADATA_FILENAME = "metadata.json"
_VERSION_LENGTH = 12


@dataclass(frozen=True)
class PreparedDataset:
    dataset: DatasetDict
    metadata: DatasetMetadata
    output_dir: Path | None


class DatasetPipeline:
    def __init__(self, config: DatasetConfig, tokenizer: TokenizerLike, seed: int) -> None:
        self._config = config
        self._tokenizer = tokenizer
        self._seed = seed

    def run(self, save: bool = True, force: bool = False) -> PreparedDataset:
        if self._config.streaming:
            raise DatasetError(
                "streaming datasets are consumed directly during training; "
                "disable dataset.streaming to prepare and cache"
            )
        source_hash = self._source_hash()
        version = self._version(source_hash)
        output_dir = Path(self._config.processed_dir) / f"{self._config.name}-{version}"

        if save and not force:
            cached = self._load_cached(output_dir)
            if cached is not None:
                return cached

        dataset = load_raw_dataset(self._config)
        if not isinstance(dataset, Dataset):
            raise DatasetError("expected a map-style dataset for preparation")

        context = PipelineContext(
            config=self._config,
            adapter=get_schema_adapter(self._config),
            tokenizer=self._tokenizer,
            seed=self._seed,
        )
        dataset = self._run_stage("validate", validate_records, dataset, context)
        dataset = self._run_stage("normalize", normalize_records, dataset, context)
        dataset = self._run_stage("clean", clean_records, dataset, context)
        dataset = self._run_stage("deduplicate", deduplicate_records, dataset, context)
        dataset = self._run_stage("language", detect_language, dataset, context)
        dataset = self._run_stage("tokenize", render_and_count_tokens, dataset, context)
        dataset_dict = split_records(dataset, context)

        metadata = DatasetMetadata(
            name=self._config.name,
            version=version,
            source_path=self._config.path,
            source_sha256=source_hash,
            record_schema=self._config.record_schema,
            num_records=sum(context.stats["splits"].values()),
            num_tokens=context.stats["num_tokens"],
            mean_tokens=round(context.stats["mean_tokens"], 2),
            max_tokens=context.stats["max_tokens"],
            language=context.stats["language"],
            created_at=datetime.now(UTC).isoformat(),
            splits=context.stats["splits"],
            records_dropped_by_cleaning=context.stats["records_dropped_by_cleaning"],
            duplicates_removed=context.stats["duplicates_removed"],
        )
        if save:
            self._persist(dataset_dict, metadata, output_dir)
            return PreparedDataset(dataset=dataset_dict, metadata=metadata, output_dir=output_dir)
        return PreparedDataset(dataset=dataset_dict, metadata=metadata, output_dir=None)

    def _source_hash(self) -> str:
        if self._config.format is DatasetFileFormat.HF_HUB:
            return sha256_text(self._config.path)
        path = Path(self._config.path)
        if not path.exists():
            raise DatasetError(f"dataset path not found: {path}")
        return sha256_dir(path) if path.is_dir() else sha256_file(path)

    def _version(self, source_hash: str) -> str:
        config_fingerprint = self._config.model_dump_json()
        return sha256_text(source_hash + config_fingerprint + str(self._seed))[:_VERSION_LENGTH]

    def _load_cached(self, output_dir: Path) -> PreparedDataset | None:
        metadata_path = output_dir / METADATA_FILENAME
        if not metadata_path.exists():
            return None
        raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        raw_metadata["record_schema"] = DatasetSchema(raw_metadata["record_schema"])
        metadata = DatasetMetadata(**raw_metadata)
        dataset_dict = load_from_disk(str(output_dir / "data"))
        if not isinstance(dataset_dict, DatasetDict):
            raise DatasetError(f"corrupted dataset cache at {output_dir}")
        logger.info("Using cached prepared dataset at {}", output_dir)
        return PreparedDataset(dataset=dataset_dict, metadata=metadata, output_dir=output_dir)

    def _persist(
        self, dataset_dict: DatasetDict, metadata: DatasetMetadata, output_dir: Path
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_dict.save_to_disk(str(output_dir / "data"))
        metadata_path = output_dir / METADATA_FILENAME
        metadata_path.write_text(
            json.dumps(asdict(metadata), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Prepared dataset saved to {}", output_dir)

    @staticmethod
    def _run_stage(
        name: str,
        stage: Callable[[Dataset, PipelineContext], Dataset],
        dataset: Dataset,
        context: PipelineContext,
    ) -> Dataset:
        start = time.perf_counter()
        result = stage(dataset, context)
        elapsed = time.perf_counter() - start
        logger.info("Stage '{}' finished in {:.2f}s ({} records)", name, elapsed, len(result))
        return result
