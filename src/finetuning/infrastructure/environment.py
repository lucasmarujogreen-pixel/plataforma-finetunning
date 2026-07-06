"""Environment snapshot for experiment reproducibility."""

import platform
import subprocess
from importlib import metadata

TRACKED_LIBRARIES = [
    "torch",
    "transformers",
    "trl",
    "peft",
    "bitsandbytes",
    "accelerate",
    "datasets",
    "safetensors",
    "mlflow",
    "pydantic",
    "hydra-core",
]


def collect_library_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for library in TRACKED_LIBRARIES:
        try:
            versions[library] = metadata.version(library)
        except metadata.PackageNotFoundError:
            versions[library] = "not-installed"
    return versions


def collect_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None
