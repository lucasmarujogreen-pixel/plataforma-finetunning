import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _load_module():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(
        "build_poc_dataset_v3", SCRIPTS_DIR / "build_poc_dataset_v3.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


module = _load_module()


def _record(analise_id: int, norma_id: int, condicoes: list[dict]) -> dict:
    return {
        "meta": {"analise_id": analise_id, "norma_id": norma_id},
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": f"Norma {norma_id}"},
            {"role": "assistant", "content": json.dumps({"condicoes": condicoes})},
        ],
    }


def test_merge_unions_items_across_analyses_by_vinculo() -> None:
    records = [
        _record(1, 10, [{"vinculo": 2, "itens": ["A > B (1) [Sim]"]}]),
        _record(2, 10, [{"vinculo": 2, "itens": ["C > D (2) [Sim]", "A > B (1) [Sim]"]}]),
        _record(3, 10, [{"vinculo": 1, "itens": ["E > F (3) [Sim]"]}]),
    ]
    merged = module.merge_norma_records(records)
    reference = json.loads(merged["messages"][-1]["content"])
    assert reference["condicoes"] == [
        {"vinculo": 1, "itens": ["E > F (3) [Sim]"]},
        {"vinculo": 2, "itens": ["A > B (1) [Sim]", "C > D (2) [Sim]"]},
    ]
    assert merged["meta"]["num_analises"] == 3
    assert merged["meta"]["analise_ids"] == [1, 2, 3]


def test_merge_deduplicates_same_item_id() -> None:
    records = [
        _record(1, 10, [{"vinculo": 2, "itens": ["A > B (1) [Sim]"]}]),
        _record(2, 10, [{"vinculo": 2, "itens": ["A > B  (1) [Sim]"]}]),
    ]
    merged = module.merge_norma_records(records)
    reference = json.loads(merged["messages"][-1]["content"])
    assert len(reference["condicoes"][0]["itens"]) == 1


def test_merge_keeps_user_message_and_system() -> None:
    records = [_record(1, 10, [{"vinculo": 2, "itens": ["A > B (1) [Sim]"]}])]
    merged = module.merge_norma_records(records)
    assert merged["messages"][0] == {"role": "system", "content": "sys"}
    assert merged["messages"][1] == {"role": "user", "content": "Norma 10"}


def test_aggregate_file_groups_by_norma(tmp_path: Path) -> None:
    records = [
        _record(1, 10, [{"vinculo": 2, "itens": ["A > B (1) [Sim]"]}]),
        _record(2, 20, [{"vinculo": 2, "itens": ["C > D (2) [Sim]"]}]),
        _record(3, 10, [{"vinculo": 2, "itens": ["C > D (2) [Sim]"]}]),
    ]
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    aggregated = module.aggregate_file(path)
    assert len(aggregated) == 2
    by_norma = {record["meta"]["norma_id"]: record for record in aggregated}
    assert by_norma[10]["meta"]["num_analises"] == 2
    assert by_norma[20]["meta"]["num_analises"] == 1
