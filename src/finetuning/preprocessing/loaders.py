"""Raw dataset loading for every supported source format."""

from pathlib import Path

from datasets import Dataset, IterableDataset, load_dataset, load_from_disk

from finetuning.core.config.schemas import DatasetConfig
from finetuning.core.enums import DatasetFileFormat
from finetuning.core.exceptions import DatasetError

_FILE_FORMAT_BUILDERS = {
    DatasetFileFormat.JSON: "json",
    DatasetFileFormat.JSONL: "json",
    DatasetFileFormat.CSV: "csv",
    DatasetFileFormat.PARQUET: "parquet",
}


def load_raw_dataset(config: DatasetConfig) -> Dataset | IterableDataset:
    """Load the configured dataset source as a map-style or streaming dataset."""
    if config.format is DatasetFileFormat.HF_HUB:
        return load_dataset(config.path, split="train", streaming=config.streaming)

    path = Path(config.path)
    if not path.exists():
        raise DatasetError(f"dataset path not found: {path}")

    if config.format is DatasetFileFormat.ARROW:
        if config.streaming:
            raise DatasetError("streaming is not supported for arrow directories")
        loaded = load_from_disk(str(path))
        if not isinstance(loaded, Dataset):
            raise DatasetError(f"expected a single split at {path}, found a dataset dict")
        return loaded

    builder = _FILE_FORMAT_BUILDERS.get(config.format)
    if builder is None:
        raise DatasetError(f"unsupported dataset format: {config.format}")
    return load_dataset(
        builder,
        data_files=str(path),
        split="train",
        streaming=config.streaming,
    )
