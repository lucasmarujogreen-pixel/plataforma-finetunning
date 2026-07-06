"""Adapters converting raw dataset records into the canonical training schema.

The canonical schema is either ``{"messages": [{"role", "content"}, ...]}`` for
conversational data or ``{"text": str}`` for plain-text corpora. New dataset
layouts plug in as new adapters registered in ``get_schema_adapter``.
"""

from abc import ABC, abstractmethod
from typing import Any

from finetuning.core.config.schemas import DatasetConfig
from finetuning.core.enums import DatasetSchema
from finetuning.core.exceptions import DatasetError

VALID_ROLES = {"system", "user", "assistant"}


class SchemaAdapter(ABC):
    """Validates and normalizes one raw record into the canonical schema."""

    @abstractmethod
    def validate(self, record: dict[str, Any]) -> str | None:
        """Return a problem description, or ``None`` when the record is valid."""

    @abstractmethod
    def normalize(self, record: dict[str, Any]) -> dict[str, Any]:
        """Convert a valid raw record into a canonical record."""


class ChatSchemaAdapter(SchemaAdapter):
    def __init__(self, messages_field: str) -> None:
        self._messages_field = messages_field

    def validate(self, record: dict[str, Any]) -> str | None:
        messages = record.get(self._messages_field)
        if not isinstance(messages, list) or not messages:
            return f"field '{self._messages_field}' must be a non-empty list"
        for position, message in enumerate(messages):
            if not isinstance(message, dict):
                return f"message {position} is not an object"
            if message.get("role") not in VALID_ROLES:
                return f"message {position} has invalid role '{message.get('role')}'"
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                return f"message {position} has empty content"
        return None

    def normalize(self, record: dict[str, Any]) -> dict[str, Any]:
        messages = [
            {"role": message["role"], "content": message["content"]}
            for message in record[self._messages_field]
        ]
        return {"messages": messages}


class AlpacaSchemaAdapter(SchemaAdapter):
    def validate(self, record: dict[str, Any]) -> str | None:
        instruction = record.get("instruction")
        output = record.get("output")
        if not isinstance(instruction, str) or not instruction.strip():
            return "field 'instruction' must be a non-empty string"
        if not isinstance(output, str) or not output.strip():
            return "field 'output' must be a non-empty string"
        return None

    def normalize(self, record: dict[str, Any]) -> dict[str, Any]:
        prompt = record["instruction"]
        extra_input = record.get("input")
        if isinstance(extra_input, str) and extra_input.strip():
            prompt = f"{prompt}\n\n{extra_input}"
        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": record["output"]},
            ]
        }


class TextSchemaAdapter(SchemaAdapter):
    def __init__(self, text_field: str) -> None:
        self._text_field = text_field

    def validate(self, record: dict[str, Any]) -> str | None:
        text = record.get(self._text_field)
        if not isinstance(text, str) or not text.strip():
            return f"field '{self._text_field}' must be a non-empty string"
        return None

    def normalize(self, record: dict[str, Any]) -> dict[str, Any]:
        return {"text": record[self._text_field]}


def get_schema_adapter(config: DatasetConfig) -> SchemaAdapter:
    if config.record_schema is DatasetSchema.CHAT:
        return ChatSchemaAdapter(config.messages_field)
    if config.record_schema is DatasetSchema.ALPACA:
        return AlpacaSchemaAdapter()
    if config.record_schema is DatasetSchema.TEXT:
        return TextSchemaAdapter(config.text_field)
    raise DatasetError(f"unsupported record schema: {config.record_schema}")
