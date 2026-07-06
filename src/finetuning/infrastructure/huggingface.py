"""Hugging Face Hub model store implementation."""

from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, LocalEntryNotFoundError
from loguru import logger

from finetuning.core.exceptions import ModelError


class HuggingFaceModelStore:
    """Stores model snapshots under a local cache directory in HF Hub layout."""

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir

    def download(self, name: str, revision: str) -> Path:
        logger.info("Downloading model snapshot: {} (revision={})", name, revision)
        try:
            path = snapshot_download(
                repo_id=name,
                revision=revision,
                cache_dir=self._cache_dir,
            )
        except (HFValidationError, OSError) as error:
            raise ModelError(f"failed to download model '{name}': {error}") from error
        logger.info("Model snapshot available at {}", path)
        return Path(path)

    def local_snapshot(self, name: str, revision: str) -> Path | None:
        try:
            path = snapshot_download(
                repo_id=name,
                revision=revision,
                cache_dir=self._cache_dir,
                local_files_only=True,
            )
        except (HFValidationError, LocalEntryNotFoundError, OSError):
            return None
        return Path(path)
