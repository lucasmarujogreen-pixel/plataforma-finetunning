"""PDA-716: kNN candidate retrieval over historical analyses.

Thin re-export shim: the implementation moved to
``finetuning.evaluation.candidate_retrieval`` so it is importable from the
installed package (needed by the reranker dataset builder and evaluator).
Kept here so ``build_poc_dataset_v2.py`` and ``tests/unit/test_knn_retrieval.py``
keep working unchanged.
"""

from finetuning.evaluation.candidate_retrieval import (
    DEFAULT_EMBEDDING_MODEL,
    FLAG_PATTERN,
    FORM_ID_PATTERN,
    EmbedFn,
    LabeledExample,
    build_neighbor_orders,
    collect_candidates,
    format_candidate_block,
    interleave_neighbors,
    load_labeled_examples,
    make_e5_embedder,
    normalize_text,
    parse_items,
)

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "FLAG_PATTERN",
    "FORM_ID_PATTERN",
    "EmbedFn",
    "LabeledExample",
    "build_neighbor_orders",
    "collect_candidates",
    "format_candidate_block",
    "interleave_neighbors",
    "load_labeled_examples",
    "make_e5_embedder",
    "normalize_text",
    "parse_items",
]
