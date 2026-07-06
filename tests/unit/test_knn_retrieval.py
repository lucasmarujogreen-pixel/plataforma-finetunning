import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _load_module(name: str):
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


knn = _load_module("knn_retrieval")


def _example(items: dict[str, str], user: str = "norma") -> "knn.LabeledExample":
    return knn.LabeledExample(
        meta={}, messages=[], query_text=user, reference={}, items=items
    )


def test_normalize_text_strips_accents_and_case() -> None:
    assert knn.normalize_text("Produção QUÍMICA") == "producao quimica"


def test_parse_items_extracts_id_path_and_strips_flag() -> None:
    reference = {
        "condicoes": [
            {"vinculo": 2, "itens": ["ATIVIDADE > Rural > Agrícola (406) [Sim]"]},
            {"vinculo": 1, "itens": ["RECURSOS DA FLORA > Manejo (570) [Sim]"]},
        ]
    }
    assert knn.parse_items(reference) == {
        "406": "ATIVIDADE > Rural > Agrícola",
        "570": "RECURSOS DA FLORA > Manejo",
    }


def test_load_labeled_examples(tmp_path: Path) -> None:
    record = {
        "meta": {"analise_id": 1},
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Norma Aérea"},
            {
                "role": "assistant",
                "content": json.dumps(
                    {"condicoes": [{"vinculo": 2, "itens": ["A > B (10) [Sim]"]}]}
                ),
            },
        ],
    }
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(record), encoding="utf-8")
    examples = knn.load_labeled_examples(path)
    assert len(examples) == 1
    assert examples[0].query_text == "norma aerea"
    assert examples[0].items == {"10": "A > B"}


def test_interleave_neighbors_dedupes_and_excludes_self() -> None:
    embedding_row = np.array([0, 1, 2, 3])
    lexical_row = np.array([1, 0, 4, 5])
    result = knn.interleave_neighbors(embedding_row, lexical_row, k_each=2, exclude_index=0)
    assert result == [1, 2, 4, 3]


def test_collect_candidates_caps_but_keeps_required() -> None:
    train = [
        _example({"1": "A", "2": "B"}),
        _example({"3": "C", "4": "D"}),
    ]
    candidates = knn.collect_candidates(
        [0, 1], train, cap=3, required_items={"9": "GOLD"}
    )
    assert "9" in candidates
    assert len(candidates) == 3


def test_collect_candidates_without_required() -> None:
    train = [_example({"1": "A"}), _example({"1": "A", "2": "B"})]
    assert knn.collect_candidates([0, 1], train, cap=10) == {"1": "A", "2": "B"}


def test_format_candidate_block_is_alphabetical() -> None:
    block = knn.format_candidate_block({"2": "ZEBRA > Item", "1": "ATIVIDADE > Item"})
    assert block.splitlines() == ["- ATIVIDADE > Item (1)", "- ZEBRA > Item (2)"]


def test_build_neighbor_orders_ranks_similar_first() -> None:
    def fake_embed(texts: list[str], prefix: str) -> np.ndarray:
        vectors = []
        for text in texts:
            vectors.append([1.0, 0.0] if "boi" in text else [0.0, 1.0])
        return np.array(vectors, dtype=np.float32)

    train_texts = ["criacao de boi", "porto maritimo", "criacao de boi e vaca"]
    query_texts = ["fiscalizacao de boi"]
    embedding_order, lexical_order = knn.build_neighbor_orders(
        train_texts, query_texts, fake_embed
    )
    assert lexical_order[0][0] in (0, 2)
    assert int(embedding_order[0][0]) in (0, 2)


def test_build_record_appends_candidates_and_meta() -> None:
    builder = _load_module("build_poc_dataset_v2")
    example = knn.LabeledExample(
        meta={"analise_id": 7},
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Norma X"},
            {"role": "assistant", "content": "{}"},
        ],
        query_text="norma x",
        reference={},
        items={},
    )
    record = builder.build_record(example, {"406": "ATIVIDADE > Rural > Agrícola"})
    assert record["meta"]["candidate_ids"] == ["406"]
    user = next(m for m in record["messages"] if m["role"] == "user")
    assert "Norma X" in user["content"]
    assert "- ATIVIDADE > Rural > Agrícola (406)" in user["content"]
    assert record["messages"][-1]["content"] == "{}"
