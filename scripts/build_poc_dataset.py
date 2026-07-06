"""Build the PDA-716 fine-tuning dataset: norma -> condições de análise.

Reads the raw Mongo export (scripts/export_greenlegis_raw.js output), converts
each record into a chat-format example and splits train/test by NormaId hash so
test normas never appear in training.

Usage: uv run python scripts/build_poc_dataset.py
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

RAW_PATH = Path("datasets/raw/greenlegis_raw.jsonl")
TRAIN_PATH = Path("datasets/raw/greenlegis_condicoes_train.jsonl")
TEST_PATH = Path("datasets/raw/greenlegis_condicoes_test.jsonl")
TEST_FRACTION = 0.10

SYSTEM_PROMPT = (
    "Você é um analista normativo da Greenlegis. Dada uma norma, identifique as "
    "condições de aplicabilidade: os itens da taxonomia de perfis de cliente aos quais "
    "a norma se vincula. Responda somente com JSON no formato "
    '{"condicoes": [{"vinculo": <int>, "itens": ["<caminho da taxonomia>", ...]}]}.'
)


def build_user_message(record: dict) -> str:
    parts = [f"Norma: {record['especie']} — {record['titulo']}".strip(" —")]
    if record["resumo"]:
        parts.append(f"Ementa: {record['resumo']}")
    return "\n".join(parts)


def build_target(record: dict) -> dict | None:
    conditions = []
    for condition in sorted(record["condicoes"], key=lambda c: c.get("sequencia") or 0):
        items = [
            item["descricao"].strip()
            for item in condition.get("itens", [])
            if item.get("marcado") and item.get("descricao")
        ]
        if not items:
            continue
        conditions.append({"vinculo": condition["vinculo"], "itens": items})
    if not conditions:
        return None
    return {"condicoes": conditions}


def split_bucket(norma_id: int) -> str:
    digest = hashlib.sha256(str(norma_id).encode()).hexdigest()
    return "test" if int(digest[:8], 16) / 0xFFFFFFFF < TEST_FRACTION else "train"


def main() -> None:
    records = []
    skipped_invalid_json = 0
    for line in RAW_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            skipped_invalid_json += 1

    stats: Counter[str] = Counter()
    train_norma_ids: set[int] = set()
    test_norma_ids: set[int] = set()
    outputs = {"train": TRAIN_PATH.open("w", encoding="utf-8"),
               "test": TEST_PATH.open("w", encoding="utf-8")}
    try:
        for record in records:
            stats["total"] += 1
            if not record["titulo"] and not record["resumo"]:
                stats["skipped_sem_norma_texto"] += 1
                continue
            target = build_target(record)
            if target is None:
                stats["skipped_sem_itens_marcados"] += 1
                continue
            bucket = split_bucket(record["norma_id"])
            (train_norma_ids if bucket == "train" else test_norma_ids).add(record["norma_id"])
            example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_message(record)},
                    {
                        "role": "assistant",
                        "content": json.dumps(target, ensure_ascii=False, sort_keys=True),
                    },
                ],
                "meta": {
                    "analise_id": record["_id"],
                    "norma_id": record["norma_id"],
                    "num_normas_vinculadas": record["num_normas"],
                },
            }
            outputs[bucket].write(json.dumps(example, ensure_ascii=False) + "\n")
            stats[bucket] += 1
    finally:
        for handle in outputs.values():
            handle.close()

    overlap = train_norma_ids & test_norma_ids
    print(f"raw lines válidas: {len(records)} (json inválido: {skipped_invalid_json})")
    for key, value in sorted(stats.items()):
        print(f"{key}: {value}")
    print(f"normas train: {len(train_norma_ids)}, normas test: {len(test_norma_ids)}, "
          f"overlap: {len(overlap)}")
    assert not overlap, "split contaminado: normas em train e test"


if __name__ == "__main__":
    main()
