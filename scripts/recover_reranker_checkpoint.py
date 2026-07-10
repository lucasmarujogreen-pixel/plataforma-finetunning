"""PDA-716: one-off recovery for a reranker run interrupted mid-training.

``ft resume-reranker`` (``TrainRerankerModel.execute(..., resume_from=...)``)
delegates checkpoint restoration to ``CrossEncoderTrainer``'s own
``resume_from_checkpoint`` machinery. That fails for this LoRA setup: the
periodic checkpoint's ``model.safetensors`` has the correct PEFT-wrapped
keys (base_layer/lora_A/lora_B/modules_to_save — verified directly), but
the Trainer's restore path expects them prefixed with the CrossEncoder's
SentenceTransformer-style module index (``0.model....``) and fails a
strict ``load_state_dict`` — the same class of key-prefix mismatch already
seen (and fixed, for the *final* save only) in ``reranker_strategies.py``.

This script sidesteps the broken restore path entirely: it builds a fresh
LoRA model exactly like a normal run, loads the checkpoint's weights
directly onto the underlying (unprefixed) model — where the keys DO match
— and continues training with a fresh optimizer/scheduler for only the
steps that were still missing. This is a warm start, not a byte-identical
resume: Adam's momentum and the exact LR position are lost, but the
learned weights (the expensive part) are not.

Usage:
  uv run python scripts/recover_reranker_checkpoint.py \
      --checkpoint runs/<run>/checkpoints/checkpoint-900
"""

import argparse
import time
from pathlib import Path

from loguru import logger
from safetensors.torch import load_file
from sentence_transformers.cross_encoder import CrossEncoderTrainer
from transformers import EarlyStoppingCallback, set_seed

from finetuning.core.enums import DeviceType
from finetuning.infrastructure.experiment_manager import ExperimentManager, load_reranker_run_config
from finetuning.monitoring.hardware import (
    detect_hardware,
    limit_vram_usage,
    resolve_attention,
    resolve_device,
    resolve_precision,
)
from finetuning.preprocessing.pair_dataset import (
    cap_docs_per_query,
    load_pair_dataset,
    to_training_columns,
)
from finetuning.training.callbacks import MetricsRecorderCallback
from finetuning.training.reranker_strategies import (
    _underlying_model,
    get_reranker_strategy,
    merge_lora_for_saving,
)
from finetuning.training.reranker_trainer_factory import (
    build_reranker_loss,
    build_reranker_training_args,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir-for-config", type=Path, default=None)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Steps to run from the loaded checkpoint (fresh optimizer/scheduler). "
        "Defaults to the config's own max_steps/num_epochs — set this to just the "
        "remaining steps to avoid redoing the whole epoch.",
    )
    args = parser.parse_args()

    run_dir_for_config = args.run_dir_for_config or args.checkpoint.parent.parent
    config = load_reranker_run_config(run_dir_for_config)
    if args.max_steps is not None:
        config = config.model_copy(
            update={"training": config.training.model_copy(update={"max_steps": args.max_steps})}
        )

    profile = detect_hardware()
    device = resolve_device(profile, config.hardware)
    precision = resolve_precision(profile, config.model.precision)
    attention = resolve_attention(profile, config.model.attention)
    limit_vram_usage(device, config.hardware)
    set_seed(config.training.seed)

    prepared = load_pair_dataset(config.dataset)
    train_dataset = to_training_columns(prepared.dataset["train"], config.dataset)
    if config.training.max_train_docs_per_query is not None:
        train_dataset = cap_docs_per_query(
            train_dataset, config.dataset, config.training.max_train_docs_per_query,
            config.training.seed,
        )
    validation_dataset = to_training_columns(prepared.dataset["validation"], config.dataset)
    has_eval_split = len(validation_dataset) > 0

    strategy = get_reranker_strategy(config.training.method)
    model = strategy.build_model(config, precision, attention, device)

    logger.info("Loading checkpoint weights directly from {}", args.checkpoint)
    state_dict = load_file(str(args.checkpoint / "model.safetensors"))
    missing, unexpected = _underlying_model(model).load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint did not load cleanly: missing={missing[:5]}... "
            f"unexpected={unexpected[:5]}..."
        )
    logger.info("Checkpoint weights loaded successfully onto the fresh LoRA model")

    manager = ExperimentManager(config.experiment.runs_dir)
    run = manager.create_run(
        config,
        prepared.metadata,
        profile,
        extras={"recovered_from_checkpoint": str(args.checkpoint)},
    )
    loss = build_reranker_loss(model, config.training.loss_type, config.training.loss_kwargs)
    training_args = build_reranker_training_args(config, run, precision, has_eval_split, [])
    tokens_per_step = (
        config.training.micro_batch_size
        * config.training.gradient_accumulation_steps
        * config.training.max_length
    )
    callbacks = [
        MetricsRecorderCallback(metrics_dir=run.metrics_dir, tokens_per_step=tokens_per_step)
    ]
    if config.training.early_stopping.enabled and has_eval_split:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.training.early_stopping.patience,
                early_stopping_threshold=config.training.early_stopping.threshold,
            )
        )
    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset if has_eval_split else None,
        loss=loss,
        callbacks=callbacks,
    )

    start = time.perf_counter()
    train_output = trainer.train()
    metrics = dict(train_output.metrics)
    if has_eval_split:
        metrics.update(trainer.evaluate())
    model = merge_lora_for_saving(model)
    model.save_pretrained(str(run.model_dir))
    total_seconds = time.perf_counter() - start
    numeric_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, int | float)}
    manager.finalize_run(run, status="completed", total_seconds=total_seconds, final_metrics=numeric_metrics)
    logger.info("Recovery training finished in {:.1f}s, artifacts at {}", total_seconds, run.root)
    print(f"RUN_DIR={run.root}")


if __name__ == "__main__":
    main()
