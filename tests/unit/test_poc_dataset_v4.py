import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _load_module(name: str):
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _load_module("build_poc_dataset_v4")
evaluator = _load_module("evaluate_poc")


def _raw(analise_id: int, norma_id: int, num_normas: int, condicoes: list[dict]) -> dict:
    return {
        "_id": analise_id,
        "num_normas": num_normas,
        "norma_id": norma_id,
        "titulo": f"lei {norma_id}",
        "resumo": "dispoe sobre teste",
        "especie": "Lei",
        "condicoes": condicoes,
    }


def _cond(vinculo: int, items: list[tuple[int, str, bool]]) -> dict:
    return {
        "vinculo": vinculo,
        "itens": [
            {"descricao": descricao, "marcado": marcado, "formulario_id": form_id}
            for form_id, descricao, marcado in items
        ],
    }


def test_load_single_norma_filters(tmp_path: Path) -> None:
    lines = [
        json.dumps(_raw(1, 10, 1, [_cond(2, [(5, "A > B (5) [Sim]", True)])])),
        json.dumps(_raw(2, 11, 3, [_cond(2, [(6, "A > C (6) [Sim]", True)])])),
        "lixo nao json",
    ]
    path = tmp_path / "raw.jsonl"
    path.write_text("\n".join(lines), encoding="utf-8")
    records = builder.load_single_norma_records(path)
    assert [record["_id"] for record in records] == [1]


def test_merge_unions_and_ignores_unmarked() -> None:
    records = [
        _raw(1, 10, 1, [_cond(2, [(5, "A > B (5) [Sim]", True), (7, "A > D (7) [Sim]", False)])]),
        _raw(2, 10, 1, [_cond(2, [(6, "A > C (6) [Sim]", True)]), _cond(1, [(8, "E > F (8) [Sim]", True)])]),
    ]
    merged = builder.merge_norma_analyses(records)
    reference = json.loads(merged["messages"][-1]["content"])
    assert reference["condicoes"] == [
        {"vinculo": 1, "itens": ["E > F (8) [Sim]"]},
        {"vinculo": 2, "itens": ["A > B (5) [Sim]", "A > C (6) [Sim]"]},
    ]
    assert merged["meta"]["num_analises"] == 2
    assert merged["meta"]["analise_ids"] == [1, 2]


def test_merge_returns_none_without_marked_items() -> None:
    records = [_raw(1, 10, 1, [_cond(2, [(5, "A > B (5) [Sim]", False)])])]
    assert builder.merge_norma_analyses(records) is None


def test_extract_paths_levels() -> None:
    target = {
        "condicoes": [
            {"vinculo": 2, "itens": ["A > B > C > D (9) [Sim]", "X (1) [Sim]"]}
        ]
    }
    assert evaluator.extract_paths(target, "leaf") == {"A > B > C > D", "X"}
    assert evaluator.extract_paths(target, "parent") == {"A > B > C", "X"}
    assert evaluator.extract_paths(target, "l2") == {"A > B", "X"}
    assert evaluator.extract_paths(target, "root") == {"A", "X"}


def test_f1_score() -> None:
    precision, recall, f1 = evaluator.f1_score(2, 1, 1)
    assert round(precision, 4) == 0.6667
    assert round(recall, 4) == 0.6667
    assert round(f1, 4) == 0.6667
    assert evaluator.f1_score(0, 0, 0) == (0.0, 0.0, 0.0)
