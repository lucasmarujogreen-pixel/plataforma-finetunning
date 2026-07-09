import pytest

from finetuning.evaluation.hierarchical_f1 import (
    HIERARCHY_LEVELS,
    extract_item_ids,
    extract_paths,
    f1_score,
    hierarchical_f1_report,
)


def _target(*items: str) -> dict:
    return {"condicoes": [{"vinculo": 1, "itens": list(items)}]}


def test_extract_item_ids_uses_numeric_id_when_present() -> None:
    target = _target("REGULARIZAÇÃO AMBIENTAL > Licença Ambiental (47)", "Varrição (12) [Sim]")

    assert extract_item_ids(target) == {"47", "12"}


def test_extract_item_ids_falls_back_to_normalized_text_without_id() -> None:
    target = _target("Item sem parênteses")

    assert extract_item_ids(target) == {"item sem parênteses"}


def test_extract_item_ids_empty_target() -> None:
    assert extract_item_ids({}) == set()
    assert extract_item_ids({"condicoes": []}) == set()


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        ("leaf", {"A > B > C"}),
        ("parent", {"A > B"}),
        ("l2", {"A > B"}),
        ("root", {"A"}),
    ],
)
def test_extract_paths_levels(level: str, expected: set[str]) -> None:
    target = _target("A > B > C (10)")

    assert extract_paths(target, level) == expected


def test_extract_paths_single_segment_parent_falls_back_to_itself() -> None:
    target = _target("A (10)")

    assert extract_paths(target, "parent") == {"A"}


def test_f1_score_all_zero_when_no_predictions_or_references() -> None:
    assert f1_score(0, 0, 0) == (0.0, 0.0, 0.0)


def test_f1_score_perfect_match() -> None:
    precision, recall, f1 = f1_score(tp=5, fp=0, fn=0)

    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)
    assert f1 == pytest.approx(1.0)


def test_f1_score_partial_match() -> None:
    precision, recall, f1 = f1_score(tp=2, fp=2, fn=2)

    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)
    assert f1 == pytest.approx(0.5)


def test_hierarchical_f1_report_perfect_predictions() -> None:
    references = [_target("A > B (1)"), _target("A > C (2)")]
    predictions = [_target("A > B (1)"), _target("A > C (2)")]

    report = hierarchical_f1_report(references, predictions)

    assert report["item_f1"] == pytest.approx(1.0)
    assert report["exact_match_rate"] == pytest.approx(1.0)
    for level in HIERARCHY_LEVELS:
        assert report["hierarchy_f1"][level] == pytest.approx(1.0)


def test_hierarchical_f1_report_none_prediction_counts_as_empty() -> None:
    references = [_target("A > B (1)")]
    predictions: list[dict | None] = [None]

    report = hierarchical_f1_report(references, predictions)

    assert report["item_f1"] == pytest.approx(0.0)
    assert report["fn"] == 1
    assert report["exact_match_rate"] == pytest.approx(0.0)


def test_hierarchical_f1_report_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        hierarchical_f1_report([_target("A (1)")], [])
