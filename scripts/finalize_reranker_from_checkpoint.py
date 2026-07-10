"""PDA-716: finalize a reranker run whose trainer.evaluate() call crashed.

Loads a checkpoint's LoRA weights directly (bypassing the training loop
entirely — no trainer.train()/evaluate() involved, so it can't hit the
CUDA "illegal memory access" that killed the recovery run's post-training
eval), merges the adapter into the base weights and saves to the run's
``model/`` folder, then marks the run as completed. Purely a save-path
recovery step: it does not retrain or re-evaluate anything.

Usage:
  uv run python scripts/finalize_reranker_from_checkpoint.py \
      --run runs/<run> --checkpoint runs/<run>/checkpoints/checkpoint-96
"""

import argparse
from pathlib import Path

from loguru import logger
from safetensors.torch import load_file

from finetuning.infrastructure.experiment_manager import ExperimentManager, load_reranker_run_config
from finetuning.monitoring.hardware import (
    detect_hardware,
    resolve_attention,
    resolve_device,
    resolve_precision,
)
from finetuning.training.reranker_strategies import (
    _underlying_model,
    get_reranker_strategy,
    merge_lora_for_saving,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    config = load_reranker_run_config(args.run)
    profile = detect_hardware()
    device = resolve_device(profile, config.hardware)
    precision = resolve_precision(profile, config.model.precision)
    attention = resolve_attention(profile, config.model.attention)

    strategy = get_reranker_strategy(config.training.method)
    model = strategy.build_model(config, precision, attention, device)

    logger.info("Loading checkpoint weights from {}", args.checkpoint)
    state_dict = load_file(str(args.checkpoint / "model.safetensors"))
    missing, unexpected = _underlying_model(model).load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint did not load cleanly: missing={missing[:5]}...")

    model = merge_lora_for_saving(model)
    model.save_pretrained(str(args.run / "model"))
    logger.info("Merged model saved to {}", args.run / "model")

    manager = ExperimentManager(config.experiment.runs_dir)
    manager.finalize_run(
        manager.load_run(args.run), status="completed", total_seconds=0.0, final_metrics=None
    )
    print(f"RUN_DIR={args.run}")


if __name__ == "__main__":
    main()
