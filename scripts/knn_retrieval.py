"""PDA-716: kNN candidate retrieval over historical analyses.

Retrieves training examples similar to a norma (e5 embeddings + TF-IDF, union)
and exposes their labeled taxonomy items as candidates. Validated offline:
candidate recall@(20+20) = 71% on the held-out test set, versus 8% for direct
norma-to-taxonomy embedding similarity.
"""

import json
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORM_ID_PATTERN = re.compile(r"\((\d+)\)")
FLAG_PATTERN = re.compile(r"\s*\[[^\]]*\]\s*$")
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

EmbedFn = Callable[[list[str], str], np.ndarray]


@dataclass(frozen=True)
class LabeledExample:
    meta: dict
    messages: list[dict]
    query_text: str
    reference: dict
    items: dict[str, str]


def normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def parse_items(reference: dict) -> dict[str, str]:
    items: dict[str, str] = {}
    for condition in reference.get("condicoes", []):
        for item in condition.get("itens", []):
            match = FORM_ID_PATTERN.search(item)
            if match is None:
                continue
            path_text = FLAG_PATTERN.sub("", item[: item.rfind("(")]).strip()
            items.setdefault(match.group(1), path_text)
    return items


def load_labeled_examples(path: Path, limit: int | None = None) -> list[LabeledExample]:
    examples = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            user_content = next(
                message["content"] for message in record["messages"] if message["role"] == "user"
            )
            reference = json.loads(record["messages"][-1]["content"])
            examples.append(
                LabeledExample(
                    meta=record.get("meta", {}),
                    messages=record["messages"],
                    query_text=normalize_text(user_content),
                    reference=reference,
                    items=parse_items(reference),
                )
            )
            if limit is not None and len(examples) >= limit:
                break
    return examples


def make_e5_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL, batch_size: int = 256) -> EmbedFn:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    if device == "cuda":
        model = model.half()

    def embed(texts: list[str], prefix: str) -> np.ndarray:
        vectors = []
        for start in range(0, len(texts), batch_size):
            batch = [prefix + text for text in texts[start : start + batch_size]]
            encoded = tokenizer(
                batch, padding=True, truncation=True, max_length=256, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                output = model(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1).to(output.last_hidden_state.dtype)
            pooled = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            vectors.append(pooled.float().cpu().numpy())
        return np.concatenate(vectors, axis=0)

    return embed


def _top_n_indices(scores: np.ndarray, top_n: int) -> np.ndarray:
    top_n = min(top_n, scores.shape[1])
    partition = np.argpartition(-scores, top_n - 1, axis=1)[:, :top_n]
    row_scores = np.take_along_axis(scores, partition, axis=1)
    return np.take_along_axis(partition, np.argsort(-row_scores, axis=1), axis=1)


def build_neighbor_orders(
    train_texts: list[str],
    query_texts: list[str],
    embed_fn: EmbedFn,
    top_n: int = 100,
    block_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize as l2_normalize

    train_matrix = embed_fn(train_texts, "passage: ")
    query_matrix = embed_fn(query_texts, "query: ")

    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    train_tfidf = l2_normalize(vectorizer.fit_transform(train_texts))
    query_tfidf = l2_normalize(vectorizer.transform(query_texts))

    embedding_blocks = []
    lexical_blocks = []
    for start in range(0, len(query_texts), block_size):
        stop = start + block_size
        embedding_scores = query_matrix[start:stop] @ train_matrix.T
        embedding_blocks.append(_top_n_indices(embedding_scores, top_n))
        lexical_scores = (query_tfidf[start:stop] @ train_tfidf.T).toarray()
        lexical_blocks.append(_top_n_indices(lexical_scores, top_n))
    return np.concatenate(embedding_blocks), np.concatenate(lexical_blocks)


def interleave_neighbors(
    embedding_row: np.ndarray,
    lexical_row: np.ndarray,
    k_each: int,
    exclude_index: int | None = None,
) -> list[int]:
    neighbors: list[int] = []
    seen: set[int] = set()
    for embedding_index, lexical_index in zip(embedding_row, lexical_row):
        for index in (int(embedding_index), int(lexical_index)):
            if index == exclude_index or index in seen:
                continue
            seen.add(index)
            neighbors.append(index)
        if len(neighbors) >= 2 * k_each:
            break
    return neighbors[: 2 * k_each]


def collect_candidates(
    neighbor_indices: list[int],
    train_examples: list[LabeledExample],
    cap: int,
    required_items: dict[str, str] | None = None,
) -> dict[str, str]:
    candidates: dict[str, str] = dict(required_items or {})
    for index in neighbor_indices:
        for item_id, path_text in train_examples[index].items.items():
            if len(candidates) >= cap and item_id not in candidates:
                continue
            candidates.setdefault(item_id, path_text)
    return candidates


def format_candidate_block(candidates: dict[str, str]) -> str:
    lines = sorted(f"- {path} ({item_id})" for item_id, path in candidates.items())
    return "\n".join(lines)
