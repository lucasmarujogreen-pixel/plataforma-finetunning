"""PDA-716: build the reranker training dataset — grouped (query, candidates, labels).

For each análise in the v5 training pool (already filtered by
``build_poc_dataset_v5.py`` to require requisitos), retrieves kNN candidates
(e5 + TF-IDF union — the same retrieval used to build the v6 causal-LM
dataset and the production kNN baseline) and labels each candidate 1.0 if it
is one of the analysis's gold condicoes, 0.0 otherwise. The gold items are
always included via ``collect_candidates(required_items=...)``, so recall of
positives is 100% by construction — the reranker's job is purely to rank
them above the kNN negatives, never to recover candidates the retrieval
missed.

Uses ``train_v5`` rather than ``train_v6`` on purpose: v6's user message
already has a "Candidatos da taxonomia" block baked in for the causal-LM
prompt, which would contaminate the reranker's query text with a stale,
redundant candidate list (this script always generates its own, fresh
candidates). v5 has the exact same 14,194-analysis universe and meta fields,
just without that block.

Output is grouped by query (listwise, one record per analysis), suitable for
``sentence_transformers.cross_encoder.losses`` ranking losses:
  {"meta": {...}, "query": "<raw user message>", "docs": [...],
   "labels": [...], "candidate_ids": [...]}

Splits the training pool itself into pair-train/pair-validation via a
norma_id hash salted independently from ``build_poc_dataset.split_bucket``
(which produced train_v5/test_v5 and train_v6/test_v6 from the same split).
``greenlegis_condicoes_test_v6.jsonl`` is never read by this script — it
stays reserved for the final POC-comparable evaluation
(``ft evaluate-reranker``).

Usage:
  uv run python scripts/build_reranker_dataset.py [--cap 40] [--k-each 20]
      [--val-fraction 0.05]
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from knn_retrieval import (
    DEFAULT_EMBEDDING_MODEL,
    LabeledExample,
    build_neighbor_orders,
    collect_candidates,
    interleave_neighbors,
    load_labeled_examples,
    make_e5_embedder,
)

TRAIN_IN = Path("datasets/raw/greenlegis_condicoes_train_v5.jsonl")
OUT_DIR = Path("datasets/processed/reranker")
TRAIN_OUT = OUT_DIR / "train_pairs_v1.jsonl"
VALIDATION_OUT = OUT_DIR / "validation_pairs_v1.jsonl"


def user_content(example: LabeledExample) -> str:
    """Raw (non-normalized) user message — this is the model input, not query_text."""
    return next(message["content"] for message in example.messages if message["role"] == "user")


def validation_bucket(norma_id: int, fraction: float) -> str:
    """Hash-based split of the train_v6 pool into pair-train/pair-validation.

    Salted independently from ``build_poc_dataset.split_bucket`` (which
    produced train_v6/test_v6 at a fixed 10%) so this never reproduces that
    boundary — ``test_v6`` stays untouched by this script.
    """
    digest = hashlib.sha256(f"reranker-val:{norma_id}".encode()).hexdigest()
    return "validation" if int(digest[:8], 16) / 0xFFFFFFFF < fraction else "train"


def build_pair_record(
    example: LabeledExample,
    neighbor_indices: list[int],
    train_examples: list[LabeledExample],
    cap: int,
) -> dict | None:
    """Build one listwise record, or None if it should be skipped."""
    if len(example.items) > cap:
        return None
    candidates = collect_candidates(
        neighbor_indices, train_examples, cap=cap, required_items=example.items
    )
    docs: list[str] = []
    labels: list[float] = []
    candidate_ids: list[str] = []
    for item_id, path_text in candidates.items():
        docs.append(path_text)
        labels.append(1.0 if item_id in example.items else 0.0)
        candidate_ids.append(item_id)
    if all(label == 1.0 for label in labels):
        return None  # listwise ranking loss needs at least one negative
    return {
        "meta": example.meta,
        "query": user_content(example),
        "docs": docs,
        "labels": labels,
        "candidate_ids": candidate_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-in", type=Path, default=TRAIN_IN)
    parser.add_argument("--cap", type=int, default=40)
    parser.add_argument("--k-each", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    args = parser.parse_args()

    examples = load_labeled_examples(args.train_in)
    print(f"análises carregadas: {len(examples)}")

    embed_fn = make_e5_embedder(args.embedding_model)
    query_texts = [example.query_text for example in examples]
    embedding_order, lexical_order = build_neighbor_orders(query_texts, query_texts, embed_fn)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train": TRAIN_OUT.open("w", encoding="utf-8"),
        "validation": VALIDATION_OUT.open("w", encoding="utf-8"),
    }
    stats: Counter[str] = Counter()
    seen: set[str] = set()
    doc_counts: list[int] = []
    positive_counts: list[int] = []
    try:
        for index, example in enumerate(examples):
            stats["total"] += 1
            neighbors = interleave_neighbors(
                embedding_order[index],
                lexical_order[index],
                k_each=args.k_each,
                exclude_index=index,
            )
            record = build_pair_record(example, neighbors, examples, args.cap)
            if record is None:
                if len(example.items) > args.cap:
                    stats["skipped_too_many_gold"] += 1
                else:
                    stats["skipped_no_negative"] += 1
                continue
            key = record["query"] + "|" + ",".join(sorted(record["candidate_ids"]))
            if key in seen:
                stats["skipped_duplicate"] += 1
                continue
            seen.add(key)
            bucket = validation_bucket(example.meta["norma_id"], args.val_fraction)
            outputs[bucket].write(json.dumps(record, ensure_ascii=False) + "\n")
            stats[bucket] += 1
            doc_counts.append(len(record["docs"]))
            positive_counts.append(sum(1 for label in record["labels"] if label == 1.0))
    finally:
        for handle in outputs.values():
            handle.close()

    print(f"train: {stats['train']} | validation: {stats['validation']}")
    print(
        f"skipped_no_negative: {stats['skipped_no_negative']} | "
        f"skipped_too_many_gold: {stats['skipped_too_many_gold']} | "
        f"skipped_duplicate: {stats['skipped_duplicate']}"
    )
    if doc_counts:
        sorted_docs = sorted(doc_counts)
        n = len(sorted_docs)
        positive_rate = sum(positive_counts) / sum(doc_counts)
        print(
            f"docs por query: p50={sorted_docs[n // 2]} "
            f"p90={sorted_docs[int(n * 0.9)]} max={sorted_docs[-1]}"
        )
        print(f"taxa de positivos: {positive_rate:.1%}")
    print(f"escrito: {TRAIN_OUT} e {VALIDATION_OUT}")


if __name__ == "__main__":
    main()
