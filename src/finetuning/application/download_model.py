"""Use case: download a base model snapshot and fingerprint it."""

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from finetuning.core.config.schemas import ModelConfig
from finetuning.core.hashing import sha256_dir
from finetuning.domain.ports import ModelStorePort


@dataclass(frozen=True)
class DownloadModelResult:
    name: str
    revision: str
    path: Path
    model_hash: str | None
    size_mb: int
    file_count: int


class DownloadModel:
    def __init__(self, store: ModelStorePort) -> None:
        self._store = store

    def execute(self, model_config: ModelConfig, compute_hash: bool = True) -> DownloadModelResult:
        path = self._store.download(model_config.name, model_config.revision)
        logger.info("Model snapshot ready at {}", path)
        files = [p for p in path.rglob("*") if p.is_file()]
        return DownloadModelResult(
            name=model_config.name,
            revision=model_config.revision,
            path=path,
            model_hash=sha256_dir(path) if compute_hash else None,
            size_mb=sum(p.stat().st_size for p in files) // (1024 * 1024),
            file_count=len(files),
        )
