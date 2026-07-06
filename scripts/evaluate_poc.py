"""PDA-716: compare model-generated conditions against human analysis on the test set.

Generates conditions for held-out normas (never seen in training), parses the JSON
output and computes item-level precision/recall/F1 plus condition-level metrics.

Usage:
  uv run python scripts/evaluate_poc.py --run runs/<run> [--limit 100] [--base-only]
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch

TEST_PATH = Path("datasets/raw/greenlegis_condicoes_test.jsonl")
FORM_ID_PATTERN = re.compile(r"\((\d+)\)")


def load_test_examples(limit: int, path: Path = TEST_PATH) -> list[dict]:
    examples = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            examples.append(json.loads(line))
            if len(examples) >= limit:
                break
    return examples


HIERARCHY_LEVELS = ("leaf", "parent", "l2", "root")


def extract_item_ids(target: dict) -> set[str]:
    ids: set[str] = set()
    for condition in target.get("condicoes", []):
        for item in condition.get("itens", []):
            match = FORM_ID_PATTERN.search(item)
            ids.add(match.group(1) if match else item.strip().lower())
    return ids


def extract_paths(target: dict, level: str) -> set[str]:
    paths: set[str] = set()
    for condition in target.get("condicoes", []):
        for item in condition.get("itens", []):
            text = item[: item.rfind("(")] if "(" in item else item
            parts = [part.strip() for part in text.split(">") if part.strip()]
            if not parts:
                continue
            if level == "leaf":
                paths.add(" > ".join(parts))
            elif level == "parent":
                paths.add(" > ".join(parts[:-1]) if len(parts) > 1 else parts[0])
            elif level == "l2":
                paths.add(" > ".join(parts[:2]))
            elif level == "root":
                paths.add(parts[0])
    return paths


def f1_score(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def parse_model_output(text: str) -> dict | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True, help="Run directory with the model.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--base-only", action="store_true", help="Evaluate the base model instead.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--test-path", type=Path, default=TEST_PATH)
    parser.add_argument("--variant-suffix", default="")
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    args = parser.parse_args()

    from finetuning.core.enums import DeviceType
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
    device = resolve_device(profile, config.hardware)
    precision = resolve_precision(profile, config.model.precision)
    attention = resolve_attention(profile, config.model.attention)
    tokenizer = load_tokenizer(config.model, config.tokenizer)

    if args.base_only:
        model = load_base_model(config, precision, attention, device)
        variant = "base" + args.variant_suffix
    else:
        model = load_trained_model(config, args.run / "model", precision, attention, device)
        variant = "finetuned" + args.variant_suffix
    model.eval()

    examples = load_test_examples(args.limit, args.test_path)
    stats: Counter[str] = Counter()
    tp = fp = fn = 0
    hier: dict[str, list[int]] = {level: [0, 0, 0] for level in HIERARCHY_LEVELS}
    outputs = []

    for example in examples:
        messages = [m for m in example["messages"] if m["role"] != "assistant"]
        reference = json.loads(example["messages"][-1]["content"])
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
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(
            generated[0][encoded.input_ids.shape[1] :], skip_special_tokens=True
        )
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

    precision_score, recall_score, f1 = f1_score(tp, fp, fn)
    report = {
        "variant": variant,
        "examples": stats["total"],
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
