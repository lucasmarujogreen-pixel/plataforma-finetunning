"""Generate a small deterministic PT-BR chat dataset for smoke testing.

Usage: uv run python scripts/generate_example_dataset.py
"""

import json
from pathlib import Path

OUTPUT_PATH = Path("datasets/raw/example_chat.jsonl")

SUBJECTS = [
    ("fine-tuning", "adaptar um modelo pré-treinado a uma tarefa específica com dados próprios"),
    ("LoRA", "treinar matrizes de baixo posto acopladas ao modelo, reduzindo memória e custo"),
    ("QLoRA", "combinar LoRA com quantização em 4 bits para treinar em GPUs menores"),
    ("quantização", "reduzir a precisão numérica dos pesos para economizar memória"),
    ("perplexidade", "medir o quão bem o modelo prevê o próximo token de um texto"),
    ("overfitting", "quando o modelo decora os dados de treino e generaliza mal"),
    ("learning rate", "controlar o tamanho do passo de atualização dos pesos"),
    ("batch size", "definir quantos exemplos são processados por passo de treino"),
    ("gradient checkpointing", "trocar computação por memória recalculando ativações"),
    ("tokenização", "converter texto em unidades numéricas que o modelo entende"),
    ("dataset", "o conjunto de exemplos usado para treinar e avaliar o modelo"),
    ("época", "uma passada completa por todos os exemplos do dataset"),
    ("checkpoint", "um snapshot dos pesos do modelo salvo durante o treino"),
    ("scheduler", "ajustar a learning rate ao longo do treinamento"),
    ("chat template", "formatar conversas no padrão esperado pelo modelo"),
]

QUESTION_TEMPLATES = [
    "O que é {subject}?",
    "Explique {subject} de forma simples.",
    "Para que serve {subject} no treinamento de modelos?",
    "Como você descreveria {subject} para um iniciante?",
    "Qual o papel de {subject} em fine-tuning de LLMs?",
    "Me dê uma definição curta de {subject}.",
    "Por que {subject} é importante?",
    "Resuma o conceito de {subject}.",
]

ANSWER_TEMPLATES = [
    "{subject} consiste em {definition}.",
    "Em resumo, {subject} serve para {definition}.",
    "De forma simples: {subject} é a técnica de {definition}.",
]


def build_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for subject_index, (subject, definition) in enumerate(SUBJECTS):
        for question_index, question_template in enumerate(QUESTION_TEMPLATES):
            answer_template = ANSWER_TEMPLATES[
                (subject_index + question_index) % len(ANSWER_TEMPLATES)
            ]
            records.append(
                {
                    "messages": [
                        {"role": "user", "content": question_template.format(subject=subject)},
                        {
                            "role": "assistant",
                            "content": answer_template.format(
                                subject=subject, definition=definition
                            ),
                        },
                    ]
                }
            )
    return records


def main() -> None:
    records = build_records()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
