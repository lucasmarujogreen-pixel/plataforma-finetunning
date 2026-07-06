from pathlib import Path

import pytest

from finetuning.application.download_model import DownloadModel
from finetuning.core.config.schemas import ModelConfig
from finetuning.infrastructure.huggingface import HuggingFaceModelStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
@pytest.mark.slow
def test_download_qwen3_snapshot() -> None:
    config = ModelConfig(name="Qwen/Qwen3-0.6B", cache_dir=PROJECT_ROOT / "models")
    result = DownloadModel(HuggingFaceModelStore(config.cache_dir)).execute(
        config, compute_hash=False
    )

    assert result.path.is_dir()
    assert (result.path / "config.json").exists()
    assert result.size_mb > 100
