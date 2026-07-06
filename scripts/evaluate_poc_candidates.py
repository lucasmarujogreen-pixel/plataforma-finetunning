"""PDA-716: candidate-grounded evaluation for condition extraction.

Retrieves the top-k taxonomy items most similar to each norma (embeddings) and
injects them as candidates in the prompt, turning open generation over 2715
labels into constrained selection. Reports retrieval ceiling (recall@k) and,
unless --retrieval-only, generation metrics for the fine-tuned or base model.

Usage:
  uv run python scripts/evaluate_poc_candidates.py --retrieval-only [--limit 100]
  uv run python scripts/evaluate_poc_candidates.py --run runs/<run> [--limit 100] [--base-only]
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from evaluate_poc import extract_item_ids, load_test_examples, parse_model_output

TRAIN_PATH = Path("datasets/raw/greenlegis_condicoes_train.jsonl")
EMBEDDINGS_CACHE = Path("datasets/cache/taxonomy_embeddings.npz")
FORM_ID_PATTERN = re.compile(r"\((\d+)\)")
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
RECALL_KS = (5, 10, 20, 50)

CANDIDATE_INSTRUCTION = (
    "\n\nCandidatos da taxonomia (use apenas itens desta lista):\n{candidates}\n"
    "Responda somente com o JSON no formato especificado, escolhendo apenas itens candidatos."
)


def build_catalog(train_path: Path) -> dict[str, str]:
    catalog: dict[str, str] = {}
    with train_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            reference = json.loads(record["messages"][-1]["content"])
            for condition in reference.get("condicoes", []):
                for item in condition.get("itens", []):
                    match = FORM_ID_PATTERN.search(item)
                    if match is None:
                        continue
                    path_text = item[: item.rfind("(")].strip()
                    catalog.setdefault(match.group(1), path_text)
    return catalog


def extract_query_text(example: dict) -> str:
    for message in example["messages"]:
        if message["role"] == "user":
            return message["content"]
    return ""


def recall_at_k(reference_ids: set[str], ranked_ids: list[str], ks: tuple[int, ...]) -> dict[int, float]:
    if not reference_ids:
        return {k: 1.0 for k in ks}
    return {
        k: len(reference_ids & set(ranked_ids[:k])) / len(reference_ids) for k in ks
    }


def build_candidate_messages(example: dict, candidates: list[tuple[str, str]]) -> list[dict]:
    lines = "\n".join(f"- {path} ({item_id})" for item_id, path in candidates)
    messages = []
    for message in example["messages"]:
        if message["role"] == "assistant":
            continue
        if message["role"] == "user":
            content = message["content"] + CANDIDATE_INSTRUCTION.format(candidates=lines)
            messages.append({"role": "user", "content": content})
        else:
            messages.append(dict(message))
    return messages


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def embed_texts(
    texts: list[str],
    tokenizer,
    model,
    device: str,
    prefix: str,
    batch_size: int = 64,
) -> np.ndarray:
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = [prefix + text for text in texts[start : start + batch_size]]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            output = model(**encoded)
        pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled, dim=-1)
        vectors.append(pooled.float().cpu().numpy())
    return np.concatenate(vectors, axis=0)


def load_catalog_embeddings(
    catalog: dict[str, str], tokenizer, model, device: str
) -> tuple[list[str], np.ndarray]:
    ids = sorted(catalog)
    if EMBEDDINGS_CACHE.exists():
        cached = np.load(EMBEDDINGS_CACHE, allow_pickle=False)
        if list(cached["ids"]) == ids:
            return ids, cached["matrix"]
    texts = [catalog[item_id] for item_id in ids]
    matrix = embed_texts(texts, tokenizer, model, device, prefix="passage: ")
    EMBEDDINGS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(EMBEDDINGS_CACHE, ids=np.array(ids), matrix=matrix)
    return ids, matrix


def rank_candidates(
    query_matrix: np.ndarray, catalog_matrix: np.ndarray, catalog_ids: list[str]
) -> list[list[str]]:
    scores = query_matrix @ catalog_matrix.T
    order = np.argsort(-scores, axis=1)
    return [[catalog_ids[index] for index in row] for row in order]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=None, help="Run directory with the model.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--base-only", action="store_true", help="Evaluate the base model instead.")
    parser.add_argument("--retrieval-only", action="store_true", help="Only report retrieval recall.")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    if not args.retrieval_only and args.run is None:
        parser.error("--run is required unless --retrieval-only")

    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)
    embedding_model = AutoModel.from_pretrained(args.embedding_model).to(device).eval()

    catalog = build_catalog(TRAIN_PATH)
    catalog_ids, catalog_matrix = load_catalog_embeddings(
        catalog, embedding_tokenizer, embedding_model, device
    )
    examples = load_test_examples(args.limit)
    queries = [extract_query_text(example) for example in examples]
    query_matrix = embed_texts(queries, embedding_tokenizer, embedding_model, device, prefix="query: ")
    rankings = rank_candidates(query_matrix, catalog_matrix, catalog_ids)

    catalog_id_set = set(catalog_ids)
    recall_totals: dict[int, float] = dict.fromkeys(RECALL_KS, 0.0)
    references = []
    in_catalog = total_reference_items = 0
    for example, ranked in zip(examples, rankings):
        reference = json.loads(example["messages"][-1]["content"])
        reference_ids = extract_item_ids(reference)
        references.append((reference, reference_ids))
        total_reference_items += len(reference_ids)
        in_catalog += len(reference_ids & catalog_id_set)
        for k, value in recall_at_k(reference_ids, ranked, RECALL_KS).items():
            recall_totals[k] += value

    retrieval_report = {
        "examples": len(examples),
        "catalog_size": len(catalog_ids),
        "reference_items_in_catalog_rate": round(in_catalog / max(total_reference_items, 1), 4),
        **{
            f"retrieval_recall@{k}": round(total / max(len(examples), 1), 4)
            for k, total in recall_totals.items()
        },
    }
    print(json.dumps(retrieval_report, ensure_ascii=False, indent=2))
    if args.retrieval_only:
        return

    del embedding_model
    torch.cuda.empty_cache()

    from finetuning.evaluation.model_loading import load_trained_model
    from finetuning.infrastructure.experiment_manager import load_run_config
    from finetuning.monitoring.hardware import (
        detect_hardware,
        resolve_attention,
        resolve_device,
        resolve_precision,
    )
    from finetuning.tokenization.loader import load_tokenizer
    from finetuning.training.strategies import load_base_model

    config = load_run_config(args.run)
    profile = detect_hardware()
    model_device = resolve_device(profile, config.hardware)
    precision = resolve_precision(profile, config.model.precision)
    attention = resolve_attention(profile, config.model.attention)
    tokenizer = load_tokenizer(config.model, config.tokenizer)

    if args.base_only:
        model = load_base_model(config, precision, attention, model_device)
        variant = "base_candidates"
    else:
        model = load_trained_model(config, args.run / "model", precision, attention, model_device)
        variant = "finetuned_candidates"
    model.eval()

    stats: Counter[str] = Counter()
    tp = fp = fn = 0
    adherent = predicted_total = 0
    outputs = []
    for example, ranked, (reference, reference_ids) in zip(examples, rankings, references):
        candidates = [(item_id, catalog[item_id]) for item_id in ranked[: args.top_k]]
        messages = build_candidate_messages(example, candidates)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        with torch.no_grad():
            generated = model.generate(
                input_ids=encoded.input_ids.to(model.device),
                attention_mask=encoded.attention_mask.to(model.device),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(
            generated[0][encoded.input_ids.shape[1] :], skip_special_tokens=True
        )
        predicted = parse_model_output(completion)
        stats["total"] += 1
        if predicted is None:
            stats["json_invalido"] += 1
            fn += len(reference_ids)
            outputs.append({"meta": example["meta"], "raw": completion[:500], "parsed": None})
            continue
        stats["json_valido"] += 1
        predicted_ids = extract_item_ids(predicted)
        candidate_ids = {item_id for item_id, _ in candidates}
        adherent += len(predicted_ids & candidate_ids)
        predicted_total += len(predicted_ids)
        tp += len(reference_ids & predicted_ids)
        fp += len(predicted_ids - reference_ids)
        fn += len(reference_ids - predicted_ids)
        if reference_ids == predicted_ids:
            stats["match_exato_itens"] += 1
        outputs.append(
            {
                "meta": example["meta"],
                "reference": reference,
                "predicted": predicted,
                "item_ids_ref": sorted(reference_ids),
                "item_ids_pred": sorted(predicted_ids),
                "candidate_ids": sorted(candidate_ids),
            }
        )

    precision_score = tp / (tp + fp) if tp + fp else 0.0
    recall_score = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision_score * recall_score / (precision_score + recall_score)
        if precision_score + recall_score
        else 0.0
    )
    report = {
        "variant": variant,
        "top_k": args.top_k,
        **retrieval_report,
        "json_valid_rate": round(stats["json_valido"] / max(stats["total"], 1), 4),
        "exact_match_rate": round(stats["match_exato_itens"] / max(stats["total"], 1), 4),
        "candidate_adherence_rate": round(adherent / max(predicted_total, 1), 4),
        "item_precision": round(precision_score, 4),
        "item_recall": round(recall_score, 4),
        "item_f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    output_dir = args.run / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"poc_report_{variant}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / f"poc_samples_{variant}.jsonl").write_text(
        "\n".join(json.dumps(o, ensure_ascii=False) for o in outputs), encoding="utf-8"
    )
    print(f"Detalhes salvos em {output_dir}/poc_samples_{variant}.jsonl")


if __name__ == "__main__":
    main()
