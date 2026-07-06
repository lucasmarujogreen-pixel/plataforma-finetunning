from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def config_dir() -> Path:
    return PROJECT_ROOT / "configs"
