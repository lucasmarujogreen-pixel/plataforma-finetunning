"""Training method strategies for the reranker vertical: LoRA and full fine-tuning.

Mirrors ``training/strategies.py`` but builds a
``sentence_transformers.cross_encoder.CrossEncoder`` (sequence classifier)
instead of an ``AutoModelForCausalLM``. Reuses ``build_peft_config`` from the
causal-LM strategies module and ``to_torch_dtype`` from ``quantization.py``.

Two things needed empirical verification beyond the causal-LM case (both
confirmed against a real train/save/reload cycle on ``BAAI/bge-reranker-v2-m3``
before finalizing this):

1. **Classifier head**: it is freshly initialized (never pretrained), so it
   must stay fully trainable rather than being LoRA-adapted — otherwise the
   model can only learn a tiny low-rank delta around a random-noise head.
   ``CLASSIFIER_HEAD_MODULES`` is excluded from ``target_modules="all-linear"``
   matching and passed as ``modules_to_save`` instead.
2. **PEFT wiring**: ``peft.get_peft_model(model.model, config)`` mutates
   ``model.model``'s submodules *in place* (replacing them with LoRA-Linear /
   ``ModulesToSaveWrapper``) and returns a *separate outer* ``PeftModel``
   object wrapping that same (now-mutated) inner model. Because the mutation
   is in place, ``model.model`` already reflects it — no reassignment to
   ``model[0].auto_model`` is needed (and doing it is actually a silent
   no-op: that attribute isn't a plain writable slot, confirmed by identity
   check — assigning to it doesn't change what it returns). The one place
   the *outer* wrapper is still needed is ``merge_and_unload()`` at save
   time, which is a ``PeftModel``-level method with no equivalent on the
   inner model alone — so ``build_model`` stashes it via
   ``object.__setattr__`` (bypassing ``nn.Module``'s auto-registration of
   ``nn.Module``-valued attributes as submodules, which would otherwise leak
   duplicate parameters into ``state_dict()``/checkpoints).
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from loguru import logger
from peft import get_peft_model
from sentence_transformers.cross_encoder import CrossEncoder

from finetuning.core.config.reranker_schemas import RerankerAppConfig
from finetuning.core.enums import (
    AttentionImplementation,
    DeviceType,
    Precision,
    RerankerTrainingMethod,
)
from finetuning.core.exceptions import TrainingError
from finetuning.training.quantization import to_torch_dtype
from finetuning.training.strategies import build_peft_config

CLASSIFIER_HEAD_MODULES = ["classifier", "pre_classifier", "score"]
_PEFT_WRAPPER_ATTR = "_peft_wrapper"


class RerankerTrainingStrategy(ABC):
    """Builds a CrossEncoder ready for training according to one method."""

    method: ClassVar[RerankerTrainingMethod]

    @abstractmethod
    def build_model(
        self,
        config: RerankerAppConfig,
        precision: Precision,
        attention: AttentionImplementation,
        device: DeviceType,
    ) -> CrossEncoder: ...


def load_base_reranker(
    config: RerankerAppConfig,
    precision: Precision,
    attention: AttentionImplementation,
    device: DeviceType,
) -> CrossEncoder:
    model_kwargs: dict[str, Any] = {"dtype": to_torch_dtype(precision)}
    if attention is not AttentionImplementation.AUTO:
        model_kwargs["attn_implementation"] = attention.value
    logger.info(
        "Loading reranker {} (precision={}, attention={}, device={})",
        config.model.name,
        precision.value,
        attention.value,
        device.value,
    )
    return CrossEncoder(
        config.model.name,
        revision=config.model.revision,
        cache_folder=str(config.model.cache_dir),
        trust_remote_code=config.model.trust_remote_code,
        device=device.value,
        model_kwargs=model_kwargs,
        num_labels=1,
        max_length=config.training.max_length,
    )


def _underlying_model(model: CrossEncoder) -> Any:
    module = model.model
    if module is None:
        raise TrainingError("CrossEncoder has no underlying transformers model")
    return module


class RerankerLoRAStrategy(RerankerTrainingStrategy):
    method = RerankerTrainingMethod.LORA

    def build_model(
        self,
        config: RerankerAppConfig,
        precision: Precision,
        attention: AttentionImplementation,
        device: DeviceType,
    ) -> CrossEncoder:
        model = load_base_reranker(config, precision, attention, device)
        peft_config = build_peft_config(
            config.lora,
            task_type="SEQ_CLS",
            exclude_modules=CLASSIFIER_HEAD_MODULES,
            modules_to_save=CLASSIFIER_HEAD_MODULES,
        )
        peft_model = get_peft_model(_underlying_model(model), peft_config)
        object.__setattr__(model, _PEFT_WRAPPER_ATTR, peft_model)
        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_model.parameters())
        logger.info(
            "trainable params: {} / {} ({:.2%})", trainable, total, trainable / max(total, 1)
        )
        return model


class RerankerFullFTStrategy(RerankerTrainingStrategy):
    method = RerankerTrainingMethod.FULL_FT

    def build_model(
        self,
        config: RerankerAppConfig,
        precision: Precision,
        attention: AttentionImplementation,
        device: DeviceType,
    ) -> CrossEncoder:
        return load_base_reranker(config, precision, attention, device)


_RERANKER_STRATEGIES: dict[RerankerTrainingMethod, type[RerankerTrainingStrategy]] = {
    RerankerLoRAStrategy.method: RerankerLoRAStrategy,
    RerankerFullFTStrategy.method: RerankerFullFTStrategy,
}


def get_reranker_strategy(method: RerankerTrainingMethod) -> RerankerTrainingStrategy:
    strategy_class = _RERANKER_STRATEGIES.get(method)
    if strategy_class is None:
        raise TrainingError(f"no reranker training strategy registered for method '{method}'")
    return strategy_class()


def merge_lora_for_saving(model: CrossEncoder) -> CrossEncoder:
    """Merge the LoRA delta into the base weights before the final save.

    Sidesteps ``CrossEncoder``'s adapter-only save/reload path entirely: the
    merged model has no adapter files, so evaluation/inference can always
    reload it with a plain ``CrossEncoder(model_dir)`` — no PEFT
    reconstruction needed. No-op if the model was never LoRA-adapted (full
    fine-tuning, or the ``_peft_wrapper`` stash from ``RerankerLoRAStrategy``
    is absent for any other reason).

    ``merge_and_unload()`` mutates the wrapped inner model
    (``model.model``) in place and returns that same object — verified the
    live ``CrossEncoder`` reflects the merge immediately, with no lora_*/
    modules_to_save keys left, and identical predict() scores before/after.
    """
    peft_wrapper = getattr(model, _PEFT_WRAPPER_ATTR, None)
    if peft_wrapper is not None:
        peft_wrapper.merge_and_unload()
    return model
