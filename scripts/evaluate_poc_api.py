"""PDA-716: evaluate an API model (prompt-enriched, no training) on the same test set.

Builds the acceptance-criteria comparison: fine-tuning versus prompt enrichment.
Each test example keeps the v6 prompt (norma + requisitos + kNN candidates) and
gains few-shot turns: the most similar solved analyses from the training set.

Usage:
  uv run --with anthropic python scripts/evaluate_poc_api.py --run runs/<run> [--limit 200]

API key resolution (no manual export needed): ``ANTHROPIC_API_KEY`` if already
set, else read from the sibling ``motor-ia`` project's ``.env``
(``AnthropicConfiguration__ApiKey__Agent``, matching motor-ia's own
convention of using the "Agent" key — not "Vera" — for batch/job workloads
like this one, to isolate quota/billing from the conversational key).

Scoring and report format mirror evaluate_poc.py so results are comparable.
"""

import argparse
import json
import os
import unicodedata
from collections import Counter
from pathlib import Path

from evaluate_poc import (
    HIERARCHY_LEVELS,
    extract_item_ids,
    extract_paths,
    f1_score,
    load_test_examples,
    parse_model_output,
)

TRAIN_PATH = Path("datasets/raw/greenlegis_condicoes_train_v6.jsonl")
TEST_PATH = Path("datasets/raw/greenlegis_condicoes_test_v6.jsonl")
MOTOR_IA_ENV_PATH = Path(__file__).resolve().parents[2] / "motor-ia" / ".env"
MOTOR_IA_ANTHROPIC_KEYS = (
    "AnthropicConfiguration__ApiKey__Agent",
    "AnthropicConfiguration__ApiKey__Vera",
)


def resolve_anthropic_api_key() -> str:
    """``ANTHROPIC_API_KEY`` if set, else the motor-ia project's own key."""
    existing = os.environ.get("ANTHROPIC_API_KEY")
    if existing:
        return existing
    if not MOTOR_IA_ENV_PATH.is_file():
        raise RuntimeError(
            f"ANTHROPIC_API_KEY not set and {MOTOR_IA_ENV_PATH} not found. "
            "Export ANTHROPIC_API_KEY or run this next to a motor-ia checkout."
        )
    values: dict[str, str] = {}
    for line in MOTOR_IA_ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    for key_name in MOTOR_IA_ANTHROPIC_KEYS:
        if values.get(key_name):
            return values[key_name]
    raise RuntimeError(
        f"None of {MOTOR_IA_ANTHROPIC_KEYS} set in {MOTOR_IA_ENV_PATH}. "
        "Export ANTHROPIC_API_KEY instead."
    )
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", (text or "").lower())
    return " ".join("".join(c for c in decomposed if not unicodedata.combining(c)).split())


def user_content(example: dict) -> str:
    return next(m["content"] for m in example["messages"] if m["role"] == "user")


def assistant_content(example: dict) -> str:
    return example["messages"][-1]["content"]


def build_few_shot_messages(example: dict, neighbors: list[dict]) -> list[dict]:
    messages = []
    for neighbor in neighbors:
        messages.append({"role": "user", "content": user_content(neighbor)})
        messages.append({"role": "assistant", "content": assistant_content(neighbor)})
    messages.append({"role": "user", "content": user_content(example)})
    return messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True, help="Run dir to store the report in.")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--few-shot", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    import numpy as np
    from anthropic import Anthropic
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize as l2_normalize

    train = load_test_examples(10**9, TRAIN_PATH)
    examples = load_test_examples(args.limit, TEST_PATH)

    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    train_matrix = l2_normalize(
        vectorizer.fit_transform([normalize_text(user_content(e)) for e in train])
    )
    test_matrix = l2_normalize(
        vectorizer.transform([normalize_text(user_content(e)) for e in examples])
    )
    order = np.argsort(-(test_matrix @ train_matrix.T).toarray(), axis=1)

    client = Anthropic(api_key=resolve_anthropic_api_key())
    system_prompt = examples[0]["messages"][0]["content"]

    stats: Counter[str] = Counter()
    tp = fp = fn = 0
    hier: dict[str, list[int]] = {level: [0, 0, 0] for level in HIERARCHY_LEVELS}
    outputs = []
    for index, example in enumerate(examples):
        neighbors = [train[int(j)] for j in order[index][: args.few_shot]]
        response = client.messages.create(
            model=args.model,
            max_tokens=args.max_new_tokens,
            system=system_prompt,
            messages=build_few_shot_messages(example, neighbors),
        )
        completion = "".join(block.text for block in response.content if block.type == "text")
        reference = json.loads(assistant_content(example))
        predicted = parse_model_output(completion)
        stats["total"] += 1
        if predicted is None:
            stats["json_invalido"] += 1
            fn += len(extract_item_ids(reference))
            for level in HIERARCHY_LEVELS:
                hier[level][2] += len(extract_paths(reference, level))
            outputs.append({"meta": example["meta"], "raw": completion[:500], "parsed": None})
            continue
        stats["json_valido"] += 1
        reference_ids = extract_item_ids(reference)
        predicted_ids = extract_item_ids(predicted)
        tp += len(reference_ids & predicted_ids)
        fp += len(predicted_ids - reference_ids)
        fn += len(reference_ids - predicted_ids)
        for level in HIERARCHY_LEVELS:
            reference_paths = extract_paths(reference, level)
            predicted_paths = extract_paths(predicted, level)
            hier[level][0] += len(reference_paths & predicted_paths)
            hier[level][1] += len(predicted_paths - reference_paths)
            hier[level][2] += len(reference_paths - predicted_paths)
        if reference_ids == predicted_ids:
            stats["match_exato_itens"] += 1
        outputs.append(
            {
                "meta": example["meta"],
                "reference": reference,
                "predicted": predicted,
                "item_ids_ref": sorted(reference_ids),
                "item_ids_pred": sorted(predicted_ids),
            }
        )
        if stats["total"] % 20 == 0:
            print(f'{stats["total"]}/{len(examples)}...')

    precision_score, recall_score, f1 = f1_score(tp, fp, fn)
    variant = f"api_{args.model.replace('/', '-')}"
    report = {
        "variant": variant,
        "examples": stats["total"],
        "few_shot": args.few_shot,
        "json_valid_rate": round(stats["json_valido"] / max(stats["total"], 1), 4),
        "exact_match_rate": round(stats["match_exato_itens"] / max(stats["total"], 1), 4),
        "item_precision": round(precision_score, 4),
        "item_recall": round(recall_score, 4),
        "item_f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "hierarchy_f1": {
            level: round(f1_score(*counts)[2], 4) for level, counts in hier.items()
        },
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
