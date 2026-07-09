"""Training method strategies: LoRA, QLoRA and full supervised fine-tuning."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from loguru import logger
from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM

from finetuning.core.config.schemas import AppConfig, LoraConfig
from finetuning.core.enums import AttentionImplementation, DeviceType, Precision, TrainingMethod
from finetuning.core.exceptions import TrainingError
from finetuning.training.quantization import build_quantization_config, to_torch_dtype


class TrainingStrategy(ABC):
    """Builds a model ready for training according to one fine-tuning method."""

    method: ClassVar[TrainingMethod]

    @abstractmethod
    def build_model(
        self,
        config: AppConfig,
        precision: Precision,
        attention: AttentionImplementation,
        device: DeviceType,
    ) -> Any: ...


def load_base_model(
    config: AppConfig,
    precision: Precision,
    attention: AttentionImplementation,
    device: DeviceType,
    quantization: Any = None,
) -> Any:
    kwargs: dict[str, Any] = {
        "revision": config.model.revision,
        "cache_dir": config.model.cache_dir,
        "trust_remote_code": config.model.trust_remote_code,
        "dtype": to_torch_dtype(precision),
    }
    if attention is not AttentionImplementation.AUTO:
        kwargs["attn_implementation"] = attention.value
    if device is DeviceType.CUDA:
        kwargs["device_map"] = {"": 0}
    if quantization is not None:
        kwargs["quantization_config"] = quantization
    logger.info(
        "Loading model {} (precision={}, attention={}, device={}, quantized={})",
        config.model.name,
        precision.value,
        attention.value,
        device.value,
        quantization is not None,
    )
    return AutoModelForCausalLM.from_pretrained(config.model.name, **kwargs)


def build_peft_config(
    lora: LoraConfig,
    task_type: str = "CAUSAL_LM",
    exclude_modules: list[str] | None = None,
    modules_to_save: list[str] | None = None,
) -> PeftLoraConfig:
    """Build a PEFT LoRA config. Shared by causal-LM strategies and the
    reranker vertical (``training/reranker_strategies.py``), which passes
    ``task_type="SEQ_CLS"`` plus ``exclude_modules``/``modules_to_save`` to
    keep the (freshly initialized) classification head fully trainable
    instead of LoRA-adapted.
    """
    return PeftLoraConfig(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=lora.target_modules,
        bias=lora.bias.value,
        task_type=task_type,
        exclude_modules=exclude_modules,
        modules_to_save=modules_to_save,
    )


class LoRAStrategy(TrainingStrategy):
    method = TrainingMethod.LORA

    def build_model(
        self,
        config: AppConfig,
        precision: Precision,
        attention: AttentionImplementation,
        device: DeviceType,
    ) -> Any:
        model = load_base_model(config, precision, attention, device)
        peft_model = get_peft_model(model, build_peft_config(config.lora))
        peft_model.print_trainable_parameters()
        return peft_model


class QLoRAStrategy(TrainingStrategy):
    method = TrainingMethod.QLORA

    def build_model(
        self,
        config: AppConfig,
        precision: Precision,
        attention: AttentionImplementation,
        device: DeviceType,
    ) -> Any:
        quantization = build_quantization_config(config.lora.quantization)
        if quantization is None:
            raise TrainingError("qlora strategy requires lora.quantization.load_in_4bit=true")
        compute_precision = config.lora.quantization.compute_precision
        model = load_base_model(config, compute_precision, attention, device, quantization)
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.training.gradient_checkpointing
        )
        peft_model = get_peft_model(model, build_peft_config(config.lora))
        peft_model.print_trainable_parameters()
        return peft_model


class FullSFTStrategy(TrainingStrategy):
    method = TrainingMethod.FULL_SFT

    def build_model(
        self,
        config: AppConfig,
        precision: Precision,
        attention: AttentionImplementation,
        device: DeviceType,
    ) -> Any:
        return load_base_model(config, precision, attention, device)


_STRATEGIES: dict[TrainingMethod, type[TrainingStrategy]] = {
    LoRAStrategy.method: LoRAStrategy,
    QLoRAStrategy.method: QLoRAStrategy,
    FullSFTStrategy.method: FullSFTStrategy,
}


def get_strategy(method: TrainingMethod) -> TrainingStrategy:
    strategy_class = _STRATEGIES.get(method)
    if strategy_class is None:
        raise TrainingError(f"no training strategy registered for method '{method}'")
    return strategy_class()
