from pathlib import Path

import pytest
from safetensors import safe_open

from finetuning.application.train_reranker_model import TrainRerankerModel
from finetuning.core.config import RerankerAppConfig, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "reranker_pairs_smoke.jsonl"


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.slow
def test_reranker_lora_smoke_training() -> None:
    config = load_config(
        PROJECT_ROOT / "configs",
        config_name="reranker",
        schema=RerankerAppConfig,
        overrides=[
            "training.max_steps=5",
            "training.eval_steps=5",
            "training.save_steps=5",
            "training.logging_steps=1",
            "hardware.num_workers=0",
            f"dataset.path={FIXTURE_PATH}",
            f"dataset.validation_path={FIXTURE_PATH}",
        ],
    )

    result = TrainRerankerModel().execute(config)

    assert result.run_dir.is_dir()
    model_dir = result.run_dir / "model"
    # The saved model is always merged (see reranker_strategies.merge_lora_for_saving),
    # so it's a plain CrossEncoder checkpoint — no adapter_config.json to check for.
    assert (model_dir / "config.json").exists()
    assert not (model_dir / "adapter_config.json").exists()
    assert (result.run_dir / "metrics" / "training_log.jsonl").exists()
    assert result.metrics.get("train_loss") is not None

    # Regression guard: merge_lora_for_saving must actually merge, not silently
    # no-op (it did, once — checked isinstance on the wrong object reference,
    # see its docstring). A real merge leaves no lora_*/modules_to_save keys.
    weights_file = model_dir / "model.safetensors"
    assert weights_file.exists()
    with safe_open(str(weights_file), framework="pt") as handle:
        keys = list(handle.keys())
    assert not any("lora_" in key or "modules_to_save" in key for key in keys)
