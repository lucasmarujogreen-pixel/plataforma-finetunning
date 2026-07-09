import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _load_module(name: str):
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


knn = _load_module("knn_retrieval")
build = _load_module("build_reranker_dataset")


def _example(
    analise_id: int, norma_id: int, items: dict[str, str], messages: list[dict] | None = None
) -> "knn.LabeledExample":
    default_messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": f"Norma {norma_id} — texto original com Acentuação"},
        {"role": "assistant", "content": "{}"},
    ]
    return knn.LabeledExample(
        meta={"analise_id": analise_id, "norma_id": norma_id},
        messages=messages or default_messages,
        query_text=f"norma {norma_id} - texto original com acentuacao",
        reference={},
        items=items,
    )


def test_user_content_returns_raw_unnormalized_text() -> None:
    example = _example(1, 10, {"1": "A > B"})

    assert build.user_content(example) == "Norma 10 — texto original com Acentuação"
    assert build.user_content(example) != example.query_text


def test_validation_bucket_is_deterministic() -> None:
    first = build.validation_bucket(12345, fraction=0.5)
    second = build.validation_bucket(12345, fraction=0.5)

    assert first == second
    assert first in {"train", "validation"}


def test_validation_bucket_respects_fraction_roughly() -> None:
    buckets = [build.validation_bucket(norma_id, fraction=0.2) for norma_id in range(2000)]
    validation_rate = buckets.count("validation") / len(buckets)

    assert 0.1 < validation_rate < 0.3


def test_build_pair_record_labels_gold_items_positive_and_neighbors_negative() -> None:
    example = _example(1, 10, items={"1": "GOLD > Item"})
    train_examples = [
        example,
        _example(2, 20, items={"2": "NEG > Item A"}),
        _example(3, 30, items={"3": "NEG > Item B"}),
    ]

    record = build.build_pair_record(
        example, neighbor_indices=[1, 2], train_examples=train_examples, cap=40
    )

    assert record is not None
    assert record["meta"] == example.meta
    assert record["query"] == build.user_content(example)
    assert "1" in record["candidate_ids"]
    label_by_id = dict(zip(record["candidate_ids"], record["labels"], strict=True))
    assert label_by_id["1"] == 1.0
    assert label_by_id["2"] == 0.0
    assert label_by_id["3"] == 0.0


def test_build_pair_record_skips_when_gold_exceeds_cap() -> None:
    example = _example(1, 10, items={str(i): f"ITEM {i}" for i in range(5)})

    record = build.build_pair_record(example, neighbor_indices=[], train_examples=[example], cap=3)

    assert record is None


def test_build_pair_record_skips_when_no_negative_available() -> None:
    example = _example(1, 10, items={"1": "GOLD > Item"})
    # No neighbors and cap large enough that required_items alone fills candidates -> no negative.
    record = build.build_pair_record(example, neighbor_indices=[], train_examples=[example], cap=40)

    assert record is None


def test_main_end_to_end_writes_train_and_validation_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [
        {
            "meta": {"analise_id": index, "norma_id": index},
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": f"Norma {index} sobre licenciamento ambiental"},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {"condicoes": [{"vinculo": 1, "itens": [f"RAMO > Item {index} ({index})"]}]}
                    ),
                },
            ],
        }
        for index in range(6)
    ]
    train_in = tmp_path / "train_v6.jsonl"
    train_in.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records), encoding="utf-8"
    )

    def fake_embedder(model_name: str, batch_size: int = 256):
        def embed(texts: list[str], prefix: str) -> np.ndarray:
            # Deterministic pseudo-embeddings: each text maps to a fixed-size
            # vector derived from its hash, good enough to exercise the kNN
            # plumbing without downloading a real model.
            rng = np.random.default_rng(abs(hash(prefix)) % (2**32))
            base = rng.standard_normal(8)
            return np.stack([base + i * 0.01 for i in range(len(texts))])

        return embed

    monkeypatch.setattr(build, "make_e5_embedder", fake_embedder)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_reranker_dataset.py",
            "--train-in",
            str(train_in),
            "--cap",
            "10",
            "--k-each",
            "3",
            "--val-fraction",
            "0.5",
        ],
    )

    build.main()

    out_dir = tmp_path / "datasets" / "processed" / "reranker"
    train_lines = (out_dir / "train_pairs_v1.jsonl").read_text(encoding="utf-8").splitlines()
    validation_lines = (
        (out_dir / "validation_pairs_v1.jsonl").read_text(encoding="utf-8").splitlines()
    )

    assert len(train_lines) + len(validation_lines) <= len(records)
    for line in train_lines + validation_lines:
        record = json.loads(line)
        assert set(record.keys()) == {"meta", "query", "docs", "labels", "candidate_ids"}
        assert 1.0 in record["labels"]
        assert 0.0 in record["labels"]
