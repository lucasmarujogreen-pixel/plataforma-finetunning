from pathlib import Path

import pytest

from finetuning.application.train_model import TrainModel
from finetuning.core.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.slow
def test_qlora_smoke_training() -> None:
    config = load_config(
        PROJECT_ROOT / "configs",
        overrides=[
            "training.max_steps=5",
            "training.eval_steps=5",
            "training.save_steps=5",
            "training.logging_steps=1",
            "hardware.num_workers=0",
        ],
    )

    result = TrainModel().execute(config)

    assert result.run_dir.is_dir()
    assert (result.run_dir / "model" / "adapter_config.json").exists()
    assert (result.run_dir / "metrics" / "training_log.jsonl").exists()
    assert result.metrics.get("train_loss") is not None
