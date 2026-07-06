"""PDA-716: build the v3 dataset — one record per norma, union of all analyses.

Diagnosis: the same norma appears in multiple analyses with disjoint condition
sets (client context absent from the input), so the v1/v2 target is
under-determined — identical inputs demand different outputs. Scoring the v1
model against the per-norma union nearly doubles precision (0.265 -> 0.476).

The v3 target aggregates every analysis of a norma into a single record:
"which conditions can this norma trigger". Items are grouped by vinculo and
sorted, making the target a consistent function of the norma text.

Usage:
  uv run python scripts/build_poc_dataset_v3.py
"""

import json
import random
from collections import defaultdict
from pathlib import Path

from knn_retrieval import FORM_ID_PATTERN

TRAIN_IN = Path("datasets/raw/greenlegis_condicoes_train.jsonl")
TEST_IN = Path("datasets/raw/greenlegis_condicoes_test.jsonl")
TRAIN_OUT = Path("datasets/raw/greenlegis_condicoes_train_v3.jsonl")
TEST_OUT = Path("datasets/raw/greenlegis_condicoes_test_v3.jsonl")


def merge_norma_records(records: list[dict]) -> dict:
    items_by_vinculo: dict[int, dict[str, str]] = defaultdict(dict)
    for record in records:
        reference = json.loads(record["messages"][-1]["content"])
        for condition in reference.get("condicoes", []):
            vinculo = condition.get("vinculo")
            for item in condition.get("itens", []):
                match = FORM_ID_PATTERN.search(item)
                if match is None:
                    continue
                items_by_vinculo[vinculo].setdefault(match.group(1), item.strip())
    condicoes = [
        {"vinculo": vinculo, "itens": sorted(items.values())}
        for vinculo, items in sorted(items_by_vinculo.items())
    ]
    base = records[0]
    messages = [dict(message) for message in base["messages"]]
    messages[-1] = {
        "role": "assistant",
        "content": json.dumps({"condicoes": condicoes}, ensure_ascii=False),
    }
    meta = dict(base.get("meta", {}))
    meta["num_analises"] = len(records)
    meta["analise_ids"] = sorted(record.get("meta", {}).get("analise_id") for record in records)
    return {"meta": meta, "messages": messages}


def aggregate_file(path: Path) -> list[dict]:
    by_norma: dict[int, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            by_norma[record["meta"]["norma_id"]].append(record)
    return [merge_norma_records(records) for _, records in sorted(by_norma.items())]


def write_records(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    for source, destination in ((TRAIN_IN, TRAIN_OUT), (TEST_IN, TEST_OUT)):
        records = aggregate_file(source)
        random.Random(42).shuffle(records)
        write_records(destination, records)
        multi = sum(1 for record in records if record["meta"]["num_analises"] > 1)
        item_counts = [
            sum(
                len(condition["itens"])
                for condition in json.loads(record["messages"][-1]["content"])["condicoes"]
            )
            for record in records
        ]
        print(
            f"{destination}: {len(records)} normas "
            f"({multi} com multiplas analises, media {sum(item_counts) / len(item_counts):.1f} itens)"
        )


if __name__ == "__main__":
    main()
