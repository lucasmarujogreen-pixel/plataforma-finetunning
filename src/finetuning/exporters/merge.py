"""Merge a LoRA adapter into the base model and save full weights."""

from pathlib import Path
from typing import Any

from loguru import logger
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from finetuning.core.config.schemas import AppConfig
from finetuning.core.enums import Precision
from finetuning.core.exceptions import ExportError
from finetuning.training.quantization import to_torch_dtype


def merge_lora_adapter(config: AppConfig, adapter_dir: Path, output_dir: Path) -> Path:
    """Merge adapter weights into the base model on CPU and save as safetensors."""
    if not (adapter_dir / "adapter_config.json").is_file():
        raise ExportError(f"no LoRA adapter found in {adapter_dir}")
    precision = config.lora.quantization.compute_precision
    if precision is Precision.AUTO:
        precision = Precision.BF16
    logger.info("Loading base model {} on CPU for merge ({})", config.model.name, precision.value)
    base: Any = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        revision=config.model.revision,
        cache_dir=config.model.cache_dir,
        trust_remote_code=config.model.trust_remote_code,
        dtype=to_torch_dtype(precision),
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    merged = model.merge_and_unload()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Merged model saved to {}", output_dir)
    return output_dir
