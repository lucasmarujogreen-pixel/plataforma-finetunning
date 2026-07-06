import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _load_module():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(
        "evaluate_poc_candidates", SCRIPTS_DIR / "evaluate_poc_candidates.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


module = _load_module()


def _write_train_fixture(tmp_path: Path) -> Path:
    records = [
        {
            "messages": [
                {"role": "user", "content": "Norma A"},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "condicoes": [
                                {
                                    "vinculo": 2,
                                    "itens": [
                                        "ATIVIDADE > Rural > Agrícola (406) [Sim]",
                                        "RECURSOS DA FLORA > Manejo (570) [Sim]",
                                    ],
                                }
                            ]
                        }
                    ),
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Norma B"},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "condicoes": [
                                {"vinculo": 1, "itens": ["ATIVIDADE > Rural > Agrícola (406) [Sim]"]}
                            ]
                        }
                    ),
                },
            ]
        },
    ]
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return path


def test_build_catalog_maps_id_to_path(tmp_path: Path) -> None:
    catalog = module.build_catalog(_write_train_fixture(tmp_path))
    assert catalog == {
        "406": "ATIVIDADE > Rural > Agrícola",
        "570": "RECURSOS DA FLORA > Manejo",
    }


def test_recall_at_k() -> None:
    ranked = ["1", "2", "3", "4"]
    result = module.recall_at_k({"2", "4"}, ranked, (1, 2, 4))
    assert result == {1: 0.0, 2: 0.5, 4: 1.0}


def test_recall_at_k_empty_reference_is_perfect() -> None:
    assert module.recall_at_k(set(), ["1"], (5,)) == {5: 1.0}


def test_build_candidate_messages_appends_candidates_and_drops_assistant() -> None:
    example = {
        "messages": [
            {"role": "system", "content": "Instrução"},
            {"role": "user", "content": "Norma X"},
            {"role": "assistant", "content": "{}"},
        ]
    }
    candidates = [("406", "ATIVIDADE > Rural > Agrícola")]
    messages = module.build_candidate_messages(example, candidates)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == "Instrução"
    assert "Norma X" in messages[1]["content"]
    assert "- ATIVIDADE > Rural > Agrícola (406)" in messages[1]["content"]
    assert example["messages"][1]["content"] == "Norma X"


def test_rank_candidates_orders_by_similarity() -> None:
    catalog_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    query_matrix = np.array([[0.9, 0.1], [0.1, 0.9]])
    rankings = module.rank_candidates(query_matrix, catalog_matrix, ["a", "b"])
    assert rankings == [["a", "b"], ["b", "a"]]


def test_extract_query_text_returns_user_content() -> None:
    example = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "norma"},
        ]
    }
    assert module.extract_query_text(example) == "norma"
