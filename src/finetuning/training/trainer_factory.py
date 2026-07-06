"""Builds the TRL SFT configuration from the validated application config."""

from trl import SFTConfig

from finetuning.core.config.schemas import AppConfig
from finetuning.core.enums import Precision
from finetuning.infrastructure.experiment_manager import ExperimentRun


def build_sft_config(
    config: AppConfig,
    run: ExperimentRun,
    precision: Precision,
    has_eval_split: bool,
    report_to: list[str],
) -> SFTConfig:
    training = config.training
    optimizer = config.optimizer
    scheduler = config.scheduler
    hardware = config.hardware
    num_workers = hardware.num_workers
    return SFTConfig(
        output_dir=str(run.checkpoints_dir),
        run_name=run.name,
        seed=training.seed,
        num_train_epochs=training.num_epochs,
        max_steps=training.max_steps,
        per_device_train_batch_size=training.micro_batch_size,
        per_device_eval_batch_size=training.micro_batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        learning_rate=optimizer.learning_rate,
        optim=optimizer.type.value,
        weight_decay=optimizer.weight_decay,
        adam_beta1=optimizer.beta1,
        adam_beta2=optimizer.beta2,
        adam_epsilon=optimizer.eps,
        max_grad_norm=optimizer.max_grad_norm,
        lr_scheduler_type=scheduler.type.value,
        warmup_ratio=scheduler.warmup_ratio,
        warmup_steps=scheduler.warmup_steps,
        max_length=training.context_length,
        packing=training.packing,
        dataset_text_field="text",
        gradient_checkpointing=training.gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if training.gradient_checkpointing else None
        ),
        bf16=precision is Precision.BF16,
        fp16=precision is Precision.FP16,
        logging_steps=training.logging_steps,
        logging_dir=str(run.logs_dir / "tensorboard"),
        eval_strategy="steps" if has_eval_split else "no",
        eval_steps=training.eval_steps if has_eval_split else None,
        save_strategy="steps",
        save_steps=training.save_steps,
        save_total_limit=training.save_total_limit,
        load_best_model_at_end=has_eval_split,
        metric_for_best_model="eval_loss" if has_eval_split else None,
        greater_is_better=False if has_eval_split else None,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=hardware.pin_memory,
        dataloader_prefetch_factor=hardware.prefetch_factor if num_workers > 0 else None,
        dataloader_persistent_workers=hardware.persistent_workers if num_workers > 0 else False,
        torch_compile=hardware.torch_compile,
        report_to=report_to,
        disable_tqdm=False,
    )
