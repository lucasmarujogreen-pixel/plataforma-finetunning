"""Hierarchical F1 scoring for the closed condicoes taxonomy (PDA-716).

Promoted from ``scripts/evaluate_poc.py`` so it is importable from the
installed package: both the causal-LM evaluation scripts and the reranker
evaluator (``application/evaluate_reranker_model.py``) need the exact same
scoring logic to stay comparable to the POC's Etapa 8 table.

A "target" is the ``{"condicoes": [{"itens": [...]}, ...]}`` JSON shape used
throughout the project (model output or ground truth). Each item string is a
taxonomy path followed by its numeric id in parentheses, e.g.
``"REGULARIZAÇÃO AMBIENTAL > Licença Ambiental (47)"``.
"""

import re
from typing import Any

FORM_ID_PATTERN = re.compile(r"\((\d+)\)")

HIERARCHY_LEVELS = ("leaf", "parent", "l2", "root")


def extract_item_ids(target: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for condition in target.get("condicoes", []):
        for item in condition.get("itens", []):
            match = FORM_ID_PATTERN.search(item)
            ids.add(match.group(1) if match else item.strip().lower())
    return ids


def extract_paths(target: dict[str, Any], level: str) -> set[str]:
    paths: set[str] = set()
    for condition in target.get("condicoes", []):
        for item in condition.get("itens", []):
            text = item[: item.rfind("(")] if "(" in item else item
            parts = [part.strip() for part in text.split(">") if part.strip()]
            if not parts:
                continue
            if level == "leaf":
                paths.add(" > ".join(parts))
            elif level == "parent":
                paths.add(" > ".join(parts[:-1]) if len(parts) > 1 else parts[0])
            elif level == "l2":
                paths.add(" > ".join(parts[:2]))
            elif level == "root":
                paths.add(parts[0])
    return paths


def f1_score(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def hierarchical_f1_report(
    references: list[dict[str, Any]], predictions: list[dict[str, Any] | None]
) -> dict[str, Any]:
    """Aggregate item-level and hierarchy-level tp/fp/fn across a test set.

    ``predictions[i] is None`` counts as an empty prediction (all reference
    items become false negatives) — mirrors how ``evaluate_poc.py`` handles
    invalid/unparseable model output.
    """
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have the same length")

    tp = fp = fn = 0
    exact_matches = 0
    hier: dict[str, list[int]] = {level: [0, 0, 0] for level in HIERARCHY_LEVELS}

    for reference, predicted in zip(references, predictions, strict=True):
        reference_ids = extract_item_ids(reference)
        predicted_ids = extract_item_ids(predicted) if predicted is not None else set()
        tp += len(reference_ids & predicted_ids)
        fp += len(predicted_ids - reference_ids)
        fn += len(reference_ids - predicted_ids)
        if predicted is not None and reference_ids == predicted_ids:
            exact_matches += 1
        for level in HIERARCHY_LEVELS:
            reference_paths = extract_paths(reference, level)
            predicted_paths = extract_paths(predicted, level) if predicted is not None else set()
            hier[level][0] += len(reference_paths & predicted_paths)
            hier[level][1] += len(predicted_paths - reference_paths)
            hier[level][2] += len(reference_paths - predicted_paths)

    precision_score, recall_score, f1 = f1_score(tp, fp, fn)
    total = len(references)
    return {
        "examples": total,
        "exact_match_rate": round(exact_matches / max(total, 1), 4),
        "item_precision": round(precision_score, 4),
        "item_recall": round(recall_score, 4),
        "item_f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "hierarchy_f1": {level: round(f1_score(*counts)[2], 4) for level, counts in hier.items()},
    }
