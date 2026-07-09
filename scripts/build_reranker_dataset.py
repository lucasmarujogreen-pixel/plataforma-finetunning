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

The train split is oversampled by ``oversample_factor()`` after dedup: 53.9%
of the 2.715 taxonomy items appear as a positive exactly once across the
whole training pool (see Etapa 9, ``docs/poc-pda-716.md``), and the trained
reranker's recall on a gold item scaled almost linearly with how often that
item was seen (0.024 never-seen vs. 0.541 seen 20+ times) — this repeats
records with rare positives (capped at 4x) so the optimizer sees them more
than once per epoch. Validation is left untouched on purpose.

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


def compute_item_frequency(records: list[dict]) -> Counter[str]:
    """Count how many training records have each taxonomy item as a positive.

    Diagnostic on the real dataset (09/07/2026): 53.9% of items appear as a
    positive exactly once, 77.3% fewer than 5 times — and the reranker's
    evaluation recall on gold items scales almost linearly with this count
    (0.024 for never-seen, 0.541 for items seen 20+ times). This long tail,
    not architecture or negative quality, is the dominant bottleneck — see
    Etapa 9 in ``docs/poc-pda-716.md``.
    """
    frequency: Counter[str] = Counter()
    for record in records:
        for candidate_id, label in zip(record["candidate_ids"], record["labels"], strict=True):
            if label == 1.0:
                frequency[candidate_id] += 1
    return frequency


def oversample_factor(record: dict, item_frequency: Counter[str]) -> int:
    """How many times to repeat a record so rare items get more gradient exposure.

    Driven by the record's rarest positive (the hardest item to learn).
    Capped at 4x: this is class reweighting, not new data — repeating a
    single exact query too many times risks memorizing it instead of
    generalizing.
    """
    positive_ids = [
        candidate_id
        for candidate_id, label in zip(record["candidate_ids"], record["labels"], strict=True)
        if label == 1.0
    ]
    rarest = min(item_frequency[candidate_id] for candidate_id in positive_ids)
    if rarest <= 1:
        return 4
    if rarest <= 4:
        return 2
    return 1


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

    stats: Counter[str] = Counter()
    seen: set[str] = set()
    records: dict[str, list[dict]] = {"train": [], "validation": []}
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
        records[bucket].append(record)
        stats[bucket] += 1

    # Oversample only the train split — validation must stay unweighted so
    # early stopping measures real generalization, not a gamed metric.
    item_frequency = compute_item_frequency(records["train"])
    oversampled_train = [
        record
        for record in records["train"]
        for _ in range(oversample_factor(record, item_frequency))
    ]
    stats["train_oversampled"] = len(oversampled_train) - len(records["train"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with TRAIN_OUT.open("w", encoding="utf-8") as handle:
        for record in oversampled_train:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with VALIDATION_OUT.open("w", encoding="utf-8") as handle:
        for record in records["validation"]:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"train: {stats['train']} (+{stats['train_oversampled']} por oversampling de itens raros)")
    print(f"validation: {stats['validation']}")
    print(
        f"skipped_no_negative: {stats['skipped_no_negative']} | "
        f"skipped_too_many_gold: {stats['skipped_too_many_gold']} | "
        f"skipped_duplicate: {stats['skipped_duplicate']}"
    )
    doc_counts = [len(record["docs"]) for record in records["train"] + records["validation"]]
    positive_counts = [
        sum(1 for label in record["labels"] if label == 1.0)
        for record in records["train"] + records["validation"]
    ]
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
