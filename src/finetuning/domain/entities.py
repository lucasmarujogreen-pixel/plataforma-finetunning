"""Domain entities shared across the platform."""

from dataclasses import dataclass, field

from finetuning.core.enums import DatasetSchema


@dataclass(frozen=True)
class DatasetMetadata:
    name: str
    version: str
    source_path: str
    source_sha256: str
    record_schema: DatasetSchema
    num_records: int
    num_tokens: int
    mean_tokens: float
    max_tokens: int
    language: str
    created_at: str
    splits: dict[str, int] = field(default_factory=dict)
    records_dropped_by_cleaning: int = 0
    duplicates_removed: int = 0
