"""Deterministic hashing utilities for reproducibility tracking."""

import hashlib
from pathlib import Path

_CHUNK_SIZE = 1024 * 1024


def sha256_text(text: str) -> str:
    """Hash a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file by streaming its content."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_dir(path: Path) -> str:
    """Hash a directory from its sorted relative file paths and file hashes."""
    digest = hashlib.sha256()
    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(file_path.relative_to(path)).encode("utf-8"))
        digest.update(sha256_file(file_path).encode("utf-8"))
    return digest.hexdigest()
