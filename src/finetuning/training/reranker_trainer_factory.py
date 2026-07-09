"""Builds the sentence-transformers CrossEncoder training config and loss.

Mirrors ``training/trainer_factory.py::build_sft_config`` but targets
``CrossEncoderTrainingArguments`` instead of ``trl.SFTConfig`` — the field
names line up closely since both ultimately extend
``transformers.TrainingArguments``, minus the causal-LM-only fields
(``max_length``/``packing``/``dataset_text_field``, which don't apply to a
sequence classifier and are instead handled by ``CrossEncoder.__init__``
via ``config.training.max_length``, see ``reranker_strategies.py``).

Deliberately always sets ``load_best_model_at_end=False``, unlike the
causal-LM config: verified end-to-end that reloading a mid-training
checkpoint reconstructs the LoRA-wrapped model fresh from disk via
``from_pretrained``, and the checkpoint's parameter names (PEFT-prefixed,
e.g. ``base_layer``/``lora_A``) don't line up with what that fresh
reconstruction expects — the load silently falls back to (partially)
random init for the mismatched names instead of raising. Early stopping
still works without this flag (``EarlyStoppingCallback`` tracks
``metric_for_best_model`` independently of whether the best checkpoint's
weights get reloaded); we just always keep training's own live, correctly
wired final model in memory for ``merge_lora_for_saving``, never reloading
from a checkpoint on disk.
"""

from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.losses import (
    BinaryCrossEntropyLoss,
    LambdaLoss,
    ListNetLoss,
    RankNetLoss,
)
from torch import nn

from finetuning.core.config.reranker_schemas import RerankerAppConfig
from finetuning.core.enums import Precision, RerankerLossType
from finetuning.core.exceptions import TrainingError
from finetuning.infrastructure.experiment_manager import ExperimentRun

_LOSSES: dict[RerankerLossType, type[nn.Module]] = {
    RerankerLossType.LAMBDA: LambdaLoss,
    RerankerLossType.LISTNET: ListNetLoss,
    RerankerLossType.RANKNET: RankNetLoss,
    RerankerLossType.BINARY_CE: BinaryCrossEntropyLoss,
}


def build_reranker_loss(
    model: CrossEncoder, loss_type: RerankerLossType, loss_kwargs: dict
) -> nn.Module:
    loss_class = _LOSSES.get(loss_type)
    if loss_class is None:
        raise TrainingError(f"no reranker loss registered for '{loss_type}'")
    return loss_class(model, **loss_kwargs)


def build_reranker_training_args(
    config: RerankerAppConfig,
    run: ExperimentRun,
    precision: Precision,
    has_eval_split: bool,
    report_to: list[str],
) -> CrossEncoderTrainingArguments:
    training = config.training
    optimizer = config.optimizer
    scheduler = config.scheduler
    hardware = config.hardware
    num_workers = hardware.num_workers
    return CrossEncoderTrainingArguments(
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
        load_best_model_at_end=False,
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
