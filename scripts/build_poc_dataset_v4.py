"""PDA-716: build the v4 dataset — clean attribution, union target per norma.

Forensics on the raw export showed two structural label problems:
1. The export attributes each analysis to Normas[0] only, so for the 46% of
   analyses linked to 2+ normas the conditions describe a group, not the norma
   in the record. v4 keeps only num_normas == 1 analyses.
2. Each analysis marks the taxonomy items of one client profile (same-norma
   analyses agree at only F1 0.081 on leaf items), so the target is the union
   of every single-norma analysis of the norma.

Split is by norma_id hash (verified: zero identical texts across sides) and
records are shuffled so any prefix is a representative sample.

Usage:
  uv run python scripts/build_poc_dataset_v4.py
"""

import json
import random
from collections import defaultdict
from pathlib import Path

from build_poc_dataset import SYSTEM_PROMPT, build_user_message, split_bucket

RAW_PATH = Path("datasets/raw/greenlegis_raw.jsonl")
TRAIN_OUT = Path("datasets/raw/greenlegis_condicoes_train_v4.jsonl")
TEST_OUT = Path("datasets/raw/greenlegis_condicoes_test_v4.jsonl")


def load_single_norma_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("num_normas") != 1:
            continue
        if not record.get("titulo") and not record.get("resumo"):
            continue
        records.append(record)
    return records


def marked_items_by_vinculo(record: dict) -> dict[int, dict[int, str]]:
    result: dict[int, dict[int, str]] = defaultdict(dict)
    for condition in record.get("condicoes", []):
        vinculo = condition.get("vinculo")
        for item in condition.get("itens", []):
            if not item.get("marcado") or not item.get("descricao"):
                continue
            form_id = item.get("formulario_id")
            if form_id is None:
                continue
            result[vinculo].setdefault(form_id, item["descricao"].strip())
    return result


def merge_norma_analyses(records: list[dict]) -> dict | None:
    items_by_vinculo: dict[int, dict[int, str]] = defaultdict(dict)
    for record in records:
        for vinculo, items in marked_items_by_vinculo(record).items():
            for form_id, descricao in items.items():
                items_by_vinculo[vinculo].setdefault(form_id, descricao)
    condicoes = [
        {"vinculo": vinculo, "itens": sorted(items.values())}
        for vinculo, items in sorted(items_by_vinculo.items())
        if items
    ]
    if not condicoes:
        return None
    base = records[0]
    return {
        "meta": {
            "norma_id": base["norma_id"],
            "num_analises": len(records),
            "analise_ids": sorted(record["_id"] for record in records),
        },
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(base)},
            {
                "role": "assistant",
                "content": json.dumps({"condicoes": condicoes}, ensure_ascii=False),
            },
        ],
    }


def main() -> None:
    records = load_single_norma_records(RAW_PATH)
    by_norma: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        by_norma[record["norma_id"]].append(record)

    buckets: dict[str, list[dict]] = {"train": [], "test": []}
    for norma_id, group in sorted(by_norma.items()):
        merged = merge_norma_analyses(group)
        if merged is None:
            continue
        buckets[split_bucket(norma_id)].append(merged)

    for bucket, path in (("train", TRAIN_OUT), ("test", TEST_OUT)):
        rows = buckets[bucket]
        random.Random(42).shuffle(rows)
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
        multi = sum(1 for row in rows if row["meta"]["num_analises"] > 1)
        item_counts = [
            sum(
                len(condition["itens"])
                for condition in json.loads(row["messages"][-1]["content"])["condicoes"]
            )
            for row in rows
        ]
        print(
            f"{path}: {len(rows)} normas ({multi} com multiplas analises, "
            f"media {sum(item_counts) / len(item_counts):.1f} itens)"
        )


if __name__ == "__main__":
    main()
