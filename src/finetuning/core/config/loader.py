"""Hydra composition combined with Pydantic validation."""

from pathlib import Path
from typing import overload

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pydantic import BaseModel, ValidationError

from finetuning.core.config.schemas import AppConfig
from finetuning.core.exceptions import ConfigurationError

DEFAULT_CONFIG_DIR = Path("configs")
DEFAULT_CONFIG_NAME = "config"


@overload
def load_config(
    config_dir: Path | str = DEFAULT_CONFIG_DIR,
    config_name: str = DEFAULT_CONFIG_NAME,
    overrides: list[str] | None = None,
) -> AppConfig: ...
@overload
def load_config[ConfigT: BaseModel](
    config_dir: Path | str = DEFAULT_CONFIG_DIR,
    config_name: str = DEFAULT_CONFIG_NAME,
    overrides: list[str] | None = None,
    *,
    schema: type[ConfigT],
) -> ConfigT: ...
def load_config(
    config_dir: Path | str = DEFAULT_CONFIG_DIR,
    config_name: str = DEFAULT_CONFIG_NAME,
    overrides: list[str] | None = None,
    schema: type[BaseModel] = AppConfig,
) -> BaseModel:
    """Compose Hydra YAML groups and validate them into a typed config.

    ``schema`` defaults to ``AppConfig`` (the causal-LM training vertical);
    pass e.g. ``schema=RerankerAppConfig`` with a matching ``config_name`` to
    compose a different vertical's config groups.
    """
    resolved_dir = Path(config_dir).resolve()
    if not resolved_dir.is_dir():
        raise ConfigurationError(f"config directory not found: {resolved_dir}")
    with initialize_config_dir(config_dir=str(resolved_dir), version_base=None):
        composed = compose(config_name=config_name, overrides=list(overrides or []))
    raw = OmegaConf.to_container(composed, resolve=True)
    try:
        return schema.model_validate(raw)
    except ValidationError as error:
        raise ConfigurationError(f"invalid configuration:\n{error}") from error
