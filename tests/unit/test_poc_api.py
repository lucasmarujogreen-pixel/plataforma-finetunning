import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _load_module():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(
        "evaluate_poc_api", SCRIPTS_DIR / "evaluate_poc_api.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


module = _load_module()


def _example(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def test_build_few_shot_messages_order_and_final_user() -> None:
    example = _example("norma alvo", "{}")
    neighbors = [_example("vizinho 1", '{"a": 1}'), _example("vizinho 2", '{"b": 2}')]
    messages = module.build_few_shot_messages(example, neighbors)
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant", "user"]
    assert messages[0]["content"] == "vizinho 1"
    assert messages[1]["content"] == '{"a": 1}'
    assert messages[-1]["content"] == "norma alvo"


def test_build_few_shot_messages_without_neighbors() -> None:
    messages = module.build_few_shot_messages(_example("alvo", "{}"), [])
    assert messages == [{"role": "user", "content": "alvo"}]


def test_normalize_text() -> None:
    assert module.normalize_text("Condição  LEGAL\n") == "condicao legal"
