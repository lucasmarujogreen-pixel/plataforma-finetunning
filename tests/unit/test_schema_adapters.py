import pytest

from finetuning.core.config.schemas import DatasetConfig
from finetuning.core.enums import DatasetFileFormat, DatasetSchema
from finetuning.preprocessing.schema_adapters import (
    AlpacaSchemaAdapter,
    ChatSchemaAdapter,
    TextSchemaAdapter,
    get_schema_adapter,
)


def make_dataset_config(schema: DatasetSchema) -> DatasetConfig:
    return DatasetConfig(
        name="test",
        path="datasets/raw/test.jsonl",
        format=DatasetFileFormat.JSONL,
        record_schema=schema,
    )


class TestChatSchemaAdapter:
    adapter = ChatSchemaAdapter(messages_field="messages")

    def test_valid_record(self) -> None:
        record = {"messages": [{"role": "user", "content": "hi"}]}

        assert self.adapter.validate(record) is None

    def test_missing_messages(self) -> None:
        assert self.adapter.validate({}) is not None

    def test_invalid_role(self) -> None:
        record = {"messages": [{"role": "robot", "content": "hi"}]}

        assert "invalid role" in str(self.adapter.validate(record))

    def test_empty_content(self) -> None:
        record = {"messages": [{"role": "user", "content": "  "}]}

        assert "empty content" in str(self.adapter.validate(record))

    def test_normalize_keeps_only_role_and_content(self) -> None:
        record = {"messages": [{"role": "user", "content": "hi", "extra": 1}], "other": True}

        normalized = self.adapter.normalize(record)

        assert normalized == {"messages": [{"role": "user", "content": "hi"}]}


class TestAlpacaSchemaAdapter:
    adapter = AlpacaSchemaAdapter()

    def test_valid_record(self) -> None:
        record = {"instruction": "Summarize", "output": "Done"}

        assert self.adapter.validate(record) is None

    def test_missing_output(self) -> None:
        assert self.adapter.validate({"instruction": "Summarize"}) is not None

    def test_normalize_builds_chat_messages(self) -> None:
        record = {"instruction": "Summarize", "input": "some text", "output": "Done"}

        normalized = self.adapter.normalize(record)

        assert normalized["messages"][0]["content"] == "Summarize\n\nsome text"
        assert normalized["messages"][1] == {"role": "assistant", "content": "Done"}


class TestTextSchemaAdapter:
    adapter = TextSchemaAdapter(text_field="text")

    def test_valid_record(self) -> None:
        assert self.adapter.validate({"text": "content"}) is None

    def test_empty_text(self) -> None:
        assert self.adapter.validate({"text": ""}) is not None

    def test_normalize(self) -> None:
        assert self.adapter.normalize({"text": "content", "extra": 1}) == {"text": "content"}


@pytest.mark.parametrize(
    ("schema", "adapter_type"),
    [
        (DatasetSchema.CHAT, ChatSchemaAdapter),
        (DatasetSchema.ALPACA, AlpacaSchemaAdapter),
        (DatasetSchema.TEXT, TextSchemaAdapter),
    ],
)
def test_get_schema_adapter(schema: DatasetSchema, adapter_type: type) -> None:
    assert isinstance(get_schema_adapter(make_dataset_config(schema)), adapter_type)
