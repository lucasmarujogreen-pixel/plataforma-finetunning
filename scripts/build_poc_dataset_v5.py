"""PDA-716: build the v5 dataset — norma + requisito texts -> condicoes da analise.

The requisito texts are the context that determines the conditions (validated:
analyses sharing the same requisito set agree at F1 0.871 versus 0.081 without
them; the copy-nearest-neighbour baseline jumps from 0.178 to 0.727). Each
record is one analysis, the well-posed unit of work.

Deduplicates identical (input, items) pairs, splits by norma_id hash and
shuffles with a fixed seed.

Usage:
  uv run python scripts/build_poc_dataset_v5.py
"""

import json
import random
from pathlib import Path

from build_poc_dataset import build_target, split_bucket

RAW_PATH = Path("datasets/raw/greenlegis_raw_v2.jsonl")
TRAIN_OUT = Path("datasets/raw/greenlegis_condicoes_train_v5.jsonl")
TEST_OUT = Path("datasets/raw/greenlegis_condicoes_test_v5.jsonl")
MAX_REQUISITOS = 8
MAX_REQUISITO_CHARS = 600

SYSTEM_PROMPT = (
    "Você é um analista normativo da Greenlegis. Dada uma norma e os requisitos legais "
    "analisados, identifique as condições de aplicabilidade: os itens da taxonomia de "
    "perfis de cliente aos quais os requisitos se vinculam. Responda somente com JSON "
    'no formato {"condicoes": [{"vinculo": <int>, "itens": ["<caminho da taxonomia>", ...]}]}.'
)


def build_user_message(record: dict) -> str:
    parts = [f"Norma: {record['especie']} — {record['titulo']}".strip(" —")]
    if record.get("resumo"):
        parts.append(f"Ementa: {record['resumo']}")
    requisitos = [
        " ".join((requisito.get("texto") or "").split())[:MAX_REQUISITO_CHARS]
        for requisito in record.get("requisitos", [])
    ]
    requisitos = [texto for texto in requisitos if texto][:MAX_REQUISITOS]
    if requisitos:
        parts.append("Requisitos analisados:")
        parts.extend(f"{index}. {texto}" for index, texto in enumerate(requisitos, start=1))
    return "\n".join(parts)


def build_example(record: dict) -> dict | None:
    target = build_target(record)
    if target is None:
        return None
    if not record.get("titulo") and not record.get("resumo"):
        return None
    return {
        "meta": {
            "analise_id": record["_id"],
            "norma_id": record["norma_id"],
            "num_normas_vinculadas": record["num_normas"],
            "num_requisitos": len(record.get("requisitos", [])),
        },
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(record)},
            {
                "role": "assistant",
                "content": json.dumps(target, ensure_ascii=False, sort_keys=True),
            },
        ],
    }


def main() -> None:
    buckets: dict[str, list[dict]] = {"train": [], "test": []}
    seen: set[str] = set()
    skipped = duplicates = 0
    for line in RAW_PATH.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        example = build_example(record)
        if example is None:
            skipped += 1
            continue
        key = example["messages"][1]["content"] + example["messages"][2]["content"]
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        buckets[split_bucket(record["norma_id"])].append(example)

    for bucket, path in (("train", TRAIN_OUT), ("test", TEST_OUT)):
        rows = buckets[bucket]
        random.Random(42).shuffle(rows)
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
        lengths = [len(row["messages"][1]["content"]) for row in rows]
        lengths.sort()
        print(
            f"{path}: {len(rows)} analises | user chars p50={lengths[len(lengths) // 2]} "
            f"p90={lengths[int(len(lengths) * 0.9)]} max={lengths[-1]}"
        )
    print(f"descartadas: {skipped} sem itens/texto | {duplicates} duplicatas exatas")


if __name__ == "__main__":
    main()
