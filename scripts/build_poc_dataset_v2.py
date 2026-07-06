"""PDA-716: build the v2 dataset with kNN candidates embedded in the prompt.

For each norma, retrieves similar historical analyses (e5 + TF-IDF union) and
appends their labeled taxonomy items as candidates to the user message, turning
open generation into constrained selection. Training examples always include
the gold items among the candidates (the model learns to select); test examples
use real retrieval only, so evaluation stays honest.

Usage:
  uv run python scripts/build_poc_dataset_v2.py [--k-each 20] [--cap 40]
"""

import argparse
import json
from pathlib import Path

from knn_retrieval import (
    LabeledExample,
    build_neighbor_orders,
    collect_candidates,
    format_candidate_block,
    interleave_neighbors,
    load_labeled_examples,
    make_e5_embedder,
)

TRAIN_IN = Path("datasets/raw/greenlegis_condicoes_train.jsonl")
TEST_IN = Path("datasets/raw/greenlegis_condicoes_test.jsonl")
TRAIN_OUT = Path("datasets/raw/greenlegis_condicoes_train_v2.jsonl")
TEST_OUT = Path("datasets/raw/greenlegis_condicoes_test_v2.jsonl")

CANDIDATE_INSTRUCTION = (
    "\n\nCandidatos da taxonomia (use apenas itens desta lista):\n{candidates}\n"
    "Responda somente com o JSON no formato especificado, escolhendo apenas itens candidatos."
)


def build_record(example: LabeledExample, candidates: dict[str, str]) -> dict:
    messages = []
    for message in example.messages:
        if message["role"] == "user":
            content = message["content"] + CANDIDATE_INSTRUCTION.format(
                candidates=format_candidate_block(candidates)
            )
            messages.append({"role": "user", "content": content})
        else:
            messages.append(dict(message))
    meta = dict(example.meta)
    meta["candidate_ids"] = sorted(candidates)
    return {"meta": meta, "messages": messages}


def write_records(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-each", type=int, default=20)
    parser.add_argument("--cap", type=int, default=40)
    parser.add_argument("--train-in", type=Path, default=TRAIN_IN)
    parser.add_argument("--test-in", type=Path, default=TEST_IN)
    parser.add_argument("--train-out", type=Path, default=TRAIN_OUT)
    parser.add_argument("--test-out", type=Path, default=TEST_OUT)
    args = parser.parse_args()

    train = load_labeled_examples(args.train_in)
    test = load_labeled_examples(args.test_in)
    print(f"train: {len(train)} | test: {len(test)} | k_each={args.k_each} cap={args.cap}")

    embed_fn = make_e5_embedder()
    train_texts = [example.query_text for example in train]
    query_texts = train_texts + [example.query_text for example in test]
    embedding_order, lexical_order = build_neighbor_orders(train_texts, query_texts, embed_fn)

    train_records = []
    for index, example in enumerate(train):
        neighbors = interleave_neighbors(
            embedding_order[index], lexical_order[index], args.k_each, exclude_index=index
        )
        candidates = collect_candidates(neighbors, train, args.cap, required_items=example.items)
        train_records.append(build_record(example, candidates))

    test_records = []
    covered = total = 0
    candidate_counts = []
    for offset, example in enumerate(test):
        row = len(train) + offset
        neighbors = interleave_neighbors(embedding_order[row], lexical_order[row], args.k_each)
        candidates = collect_candidates(neighbors, train, args.cap)
        candidate_counts.append(len(candidates))
        covered += len(set(example.items) & set(candidates))
        total += len(example.items)
        test_records.append(build_record(example, candidates))

    write_records(args.train_out, train_records)
    write_records(args.test_out, test_records)

    average_candidates = sum(candidate_counts) / max(len(candidate_counts), 1)
    print(f"test candidate recall (teto): {covered / max(total, 1):.1%}")
    print(f"test avg candidates: {average_candidates:.0f}")
    print(f"escrito: {args.train_out} ({len(train_records)}) e {args.test_out} ({len(test_records)})")


if __name__ == "__main__":
    main()
