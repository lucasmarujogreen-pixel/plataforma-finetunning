"""Interfaces (ports) implemented by the infrastructure layer."""

from pathlib import Path
from typing import Protocol


class ModelStorePort(Protocol):
    """Downloads and locates model snapshots on the local filesystem."""

    def download(self, name: str, revision: str) -> Path: ...

    def local_snapshot(self, name: str, revision: str) -> Path | None: ...
