"""Use case: evaluate a trained reranker on the POC's canonical test set.

Reuses the same test file, sample limit and kNN candidate retrieval as the
causal-LM evaluation scripts (``scripts/evaluate_poc*.py``), so results are
directly comparable to the POC's Etapa 8 table (kNN 0.559, API Haiku 0.500,
fine-tuned v6 0.457, retrieval recall ceiling 0.737 — see
``docs/poc-pda-716.md``).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finetuning.core.config.reranker_schemas import SelectionConfig
from finetuning.evaluation.candidate_retrieval import (
    build_neighbor_orders,
    collect_candidates,
    interleave_neighbors,
    load_labeled_examples,
    make_e5_embedder,
    normalize_text,
)
from finetuning.evaluation.hierarchical_f1 import extract_item_ids, hierarchical_f1_report
from finetuning.evaluation.reranker_model_loading import load_trained_reranker
from finetuning.infrastructure.experiment_manager import load_reranker_run_config
from finetuning.monitoring.hardware import (
    detect_hardware,
    resolve_attention,
    resolve_device,
    resolve_precision,
)

_Ranked = tuple[str, str, float]


@dataclass(frozen=True)
class EvaluateRerankerModelResult:
    report: dict[str, Any]
    report_path: Path
    samples_path: Path


def _load_test_examples(path: Path, limit: int) -> list[dict]:
    examples = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            examples.append(json.loads(line))
            if len(examples) >= limit:
                break
    return examples


def _user_content(example: dict) -> str:
    return next(message["content"] for message in example["messages"] if message["role"] == "user")


def _recall_at_k(reference_ids: set[str], ranked_ids: list[str], k: int) -> float:
    if not reference_ids:
        return 1.0
    return len(reference_ids & set(ranked_ids[:k])) / len(reference_ids)


def _select(ranked: list[_Ranked], selection: SelectionConfig) -> list[_Ranked]:
    if selection.strategy == "top_k":
        return ranked[: selection.top_k]
    return [item for item in ranked if item[2] >= selection.threshold]


class EvaluateRerankerModel:
    def execute(self, run_dir: Path) -> EvaluateRerankerModelResult:
        config = load_reranker_run_config(run_dir)
        evaluation = config.evaluation
        profile = detect_hardware()
        device = resolve_device(profile, config.hardware)
        precision = resolve_precision(profile, config.model.precision)
        attention = resolve_attention(profile, config.model.attention)
        model = load_trained_reranker(config, run_dir / "model", precision, attention, device)

        pool = load_labeled_examples(evaluation.train_path)
        pool_texts = [example.query_text for example in pool]
        embed_fn = make_e5_embedder()

        test_examples = _load_test_examples(evaluation.test_path, evaluation.limit)
        query_texts = [normalize_text(_user_content(example)) for example in test_examples]
        embedding_order, lexical_order = build_neighbor_orders(pool_texts, query_texts, embed_fn)

        references: list[dict[str, Any]] = []
        predictions: list[dict[str, Any] | None] = []
        recalls: list[float] = []
        for index, example in enumerate(test_examples):
            reference = json.loads(example["messages"][-1]["content"])
            references.append(reference)
            reference_ids = extract_item_ids(reference)

            neighbors = interleave_neighbors(
                embedding_order[index], lexical_order[index], k_each=20
            )
            candidates = collect_candidates(
                neighbors, pool, cap=evaluation.knn_candidates_per_query
            )
            recalls.append(
                _recall_at_k(
                    reference_ids, list(candidates.keys()), evaluation.knn_candidates_per_query
                )
            )
            if not candidates:
                predictions.append(None)
                continue

            query = _user_content(example)
            pairs = [(query, path_text) for path_text in candidates.values()]
            scores = model.predict(pairs)  # type: ignore[arg-type]
            ranked: list[_Ranked] = sorted(
                zip(candidates.keys(), candidates.values(), scores, strict=True),
                key=lambda row: -row[2],
            )
            selected = _select(ranked, evaluation.selection)
            predictions.append(
                {
                    "condicoes": [
                        {
                            "vinculo": 1,
                            "itens": [f"{path} ({item_id})" for item_id, path, _ in selected],
                        }
                    ]
                }
            )

        report = hierarchical_f1_report(references, predictions)
        report["variant"] = "reranker"
        report["retrieval_recall_at_k"] = round(sum(recalls) / max(len(recalls), 1), 4)

        output_dir = run_dir / "evaluation"
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "poc_report_reranker.json"
        samples_path = output_dir / "poc_samples_reranker.jsonl"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        samples_path.write_text(
            "\n".join(
                json.dumps({"reference": reference, "predicted": predicted}, ensure_ascii=False)
                for reference, predicted in zip(references, predictions, strict=True)
            ),
            encoding="utf-8",
        )
        return EvaluateRerankerModelResult(
            report=report, report_path=report_path, samples_path=samples_path
        )
