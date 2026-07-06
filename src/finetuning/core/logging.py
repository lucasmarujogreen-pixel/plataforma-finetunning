"""Structured logging setup built on loguru."""

import sys
from pathlib import Path

from loguru import logger

CONSOLE_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan> - <level>{message}</level>"
)
FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure the global logger with a console sink and an optional file sink."""
    logger.remove()
    logger.add(sys.stderr, level=level.upper(), format=CONSOLE_FORMAT, colorize=True)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level="DEBUG",
            format=FILE_FORMAT,
            rotation="50 MB",
            retention=10,
            encoding="utf-8",
        )
