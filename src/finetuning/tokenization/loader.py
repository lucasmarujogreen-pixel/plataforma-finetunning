"""Tokenizer loading applying platform tokenizer settings."""

from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from finetuning.core.config.schemas import ModelConfig, TokenizerConfig
from finetuning.core.exceptions import ModelError


def load_tokenizer(
    model_config: ModelConfig, tokenizer_config: TokenizerConfig
) -> PreTrainedTokenizerBase:
    """Load the model tokenizer from the local cache, downloading it if needed."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.name,
            revision=model_config.revision,
            cache_dir=Path(model_config.cache_dir),
            trust_remote_code=model_config.trust_remote_code,
        )
    except (OSError, ValueError) as error:
        raise ModelError(f"failed to load tokenizer for '{model_config.name}': {error}") from error
    tokenizer.padding_side = tokenizer_config.padding_side.value
    if tokenizer_config.chat_template is not None:
        tokenizer.chat_template = tokenizer_config.chat_template
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
