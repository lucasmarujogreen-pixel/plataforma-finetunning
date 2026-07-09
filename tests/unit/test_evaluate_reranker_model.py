import json
from pathlib import Path

from finetuning.application.evaluate_reranker_model import (
    _load_test_examples,
    _recall_at_k,
    _select,
    _user_content,
)
from finetuning.core.config.reranker_schemas import SelectionConfig


def test_user_content_extracts_the_user_message() -> None:
    example = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "the query"},
            {"role": "assistant", "content": "{}"},
        ]
    }

    assert _user_content(example) == "the query"


def test_recall_at_k_all_found() -> None:
    assert _recall_at_k({"1", "2"}, ["1", "2", "3"], k=3) == 1.0


def test_recall_at_k_partial() -> None:
    assert _recall_at_k({"1", "2"}, ["3", "1"], k=2) == 0.5


def test_recall_at_k_beyond_k_does_not_count() -> None:
    assert _recall_at_k({"1", "2"}, ["1", "3", "2"], k=1) == 0.5


def test_recall_at_k_no_reference_items_is_vacuously_perfect() -> None:
    assert _recall_at_k(set(), ["1"], k=5) == 1.0


def test_select_top_k_takes_the_first_n() -> None:
    ranked = [("1", "A", 0.9), ("2", "B", 0.5), ("3", "C", 0.1)]
    selection = SelectionConfig(strategy="top_k", top_k=2)

    assert _select(ranked, selection) == ranked[:2]


def test_select_threshold_filters_by_score() -> None:
    ranked = [("1", "A", 0.9), ("2", "B", 0.5), ("3", "C", 0.1)]
    selection = SelectionConfig(strategy="threshold", threshold=0.5)

    assert _select(ranked, selection) == [("1", "A", 0.9), ("2", "B", 0.5)]


def test_load_test_examples_respects_limit(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    path.write_text("\n".join(json.dumps({"id": i}) for i in range(5)), encoding="utf-8")

    examples = _load_test_examples(path, limit=3)

    assert examples == [{"id": 0}, {"id": 1}, {"id": 2}]
