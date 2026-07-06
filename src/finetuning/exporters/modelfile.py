"""Ollama Modelfile generation."""

from pathlib import Path

from finetuning.core.config.schemas import OllamaConfig


def build_modelfile(ollama: OllamaConfig, gguf_filename: str) -> str:
    lines = [
        f"FROM ./{gguf_filename}",
        f"PARAMETER temperature {ollama.temperature}",
        f"PARAMETER top_p {ollama.top_p}",
        f"PARAMETER num_ctx {ollama.num_ctx}",
    ]
    if ollama.system_prompt:
        lines.append(f'SYSTEM """{ollama.system_prompt}"""')
    return "\n".join(lines) + "\n"


def write_modelfile(ollama: OllamaConfig, gguf_path: Path) -> Path:
    modelfile_path = gguf_path.parent / "Modelfile"
    modelfile_path.write_text(build_modelfile(ollama, gguf_path.name), encoding="utf-8")
    return modelfile_path
