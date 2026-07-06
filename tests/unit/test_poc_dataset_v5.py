import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _load_module():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(
        "build_poc_dataset_v5", SCRIPTS_DIR / "build_poc_dataset_v5.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


module = _load_module()


def _record(**overrides) -> dict:
    record = {
        "_id": 1,
        "num_normas": 1,
        "norma_id": 10,
        "titulo": "lei 123",
        "resumo": "dispoe sobre teste",
        "especie": "Lei",
        "requisitos": [{"id": 5, "texto": "Manter registro de logradouros"}],
        "condicoes": [
            {
                "sequencia": 1,
                "vinculo": 2,
                "itens": [
                    {"descricao": "A > B (5) [Sim]", "marcado": True, "formulario_id": 5}
                ],
            }
        ],
    }
    record.update(overrides)
    return record


def test_user_message_includes_norma_and_requisitos() -> None:
    message = module.build_user_message(_record())
    assert "Norma: Lei — lei 123" in message
    assert "Ementa: dispoe sobre teste" in message
    assert "Requisitos analisados:" in message
    assert "1. Manter registro de logradouros" in message


def test_user_message_caps_and_normalizes_requisitos() -> None:
    requisitos = [{"id": i, "texto": "x" * 2000} for i in range(12)]
    requisitos.append({"id": 99, "texto": "   "})
    message = module.build_user_message(_record(requisitos=requisitos))
    lines = [line for line in message.splitlines() if line[:1].isdigit()]
    assert len(lines) == module.MAX_REQUISITOS
    assert all(len(line) <= module.MAX_REQUISITO_CHARS + 4 for line in lines)


def test_user_message_collapses_whitespace() -> None:
    record = _record(requisitos=[{"id": 5, "texto": "linha um\n\n\tlinha  dois"}])
    assert "1. linha um linha dois" in module.build_user_message(record)


def test_build_example_structure_and_meta() -> None:
    example = module.build_example(_record())
    assert example["meta"] == {
        "analise_id": 1,
        "norma_id": 10,
        "num_normas_vinculadas": 1,
        "num_requisitos": 1,
    }
    target = json.loads(example["messages"][2]["content"])
    assert target == {"condicoes": [{"itens": ["A > B (5) [Sim]"], "vinculo": 2}]}


def test_build_example_none_without_marked_items() -> None:
    record = _record(
        condicoes=[
            {
                "sequencia": 1,
                "vinculo": 2,
                "itens": [{"descricao": "A (5) [Sim]", "marcado": False, "formulario_id": 5}],
            }
        ]
    )
    assert module.build_example(record) is None
