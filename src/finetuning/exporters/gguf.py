"""GGUF conversion and quantization via llama.cpp."""

import subprocess
import sys
from pathlib import Path

from loguru import logger

from finetuning.core.enums import GGUFQuantization
from finetuning.core.exceptions import ExportError

LLAMACPP_DIR = Path("third_party/llama.cpp")
SETUP_HINT = "run 'bash scripts/setup_llamacpp.sh' to install llama.cpp"


def find_convert_script(llamacpp_dir: Path = LLAMACPP_DIR) -> Path | None:
    script = llamacpp_dir / "convert_hf_to_gguf.py"
    return script if script.is_file() else None


def find_quantize_binary(llamacpp_dir: Path = LLAMACPP_DIR) -> Path | None:
    for candidate in (
        llamacpp_dir / "build" / "bin" / "llama-quantize",
        llamacpp_dir / "llama-quantize",
    ):
        if candidate.is_file():
            return candidate
    return None


def _run(command: list[str], step: str) -> None:
    logger.info("Running {}: {}", step, " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise ExportError(f"{step} failed:\n{result.stderr[-2000:]}")


def convert_to_gguf(
    merged_model_dir: Path,
    output_path: Path,
    quantization: GGUFQuantization,
    llamacpp_dir: Path = LLAMACPP_DIR,
) -> Path:
    """Convert a merged HF model directory to a (quantized) GGUF file."""
    convert_script = find_convert_script(llamacpp_dir)
    if convert_script is None:
        raise ExportError(f"llama.cpp convert script not found; {SETUP_HINT}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    f16_path = (
        output_path
        if quantization is GGUFQuantization.F16
        else output_path.with_name(f"{output_path.stem}-f16.gguf")
    )
    _run(
        [
            sys.executable,
            str(convert_script),
            str(merged_model_dir),
            "--outfile",
            str(f16_path),
            "--outtype",
            "f16",
        ],
        "GGUF conversion",
    )
    if quantization is GGUFQuantization.F16:
        logger.info("GGUF written to {}", output_path)
        return output_path

    quantize_binary = find_quantize_binary(llamacpp_dir)
    if quantize_binary is None:
        raise ExportError(f"llama-quantize binary not found; {SETUP_HINT}")
    _run(
        [str(quantize_binary), str(f16_path), str(output_path), quantization.value.upper()],
        "GGUF quantization",
    )
    f16_path.unlink(missing_ok=True)
    logger.info("Quantized GGUF written to {}", output_path)
    return output_path
