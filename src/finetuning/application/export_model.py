"""Use case: export a trained run as adapter, merged safetensors, GGUF and Modelfile."""

import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from finetuning.core.enums import ExportFormat
from finetuning.core.exceptions import ExportError
from finetuning.exporters.gguf import convert_to_gguf
from finetuning.exporters.merge import merge_lora_adapter
from finetuning.exporters.modelfile import write_modelfile
from finetuning.infrastructure.experiment_manager import load_run_config

EXPORT_SUMMARY_FILENAME = "export.json"


@dataclass(frozen=True)
class ExportResult:
    run_name: str
    artifacts: dict[str, str] = field(default_factory=dict)


class ExportModel:
    def execute(self, run_dir: Path, formats: list[ExportFormat] | None = None) -> ExportResult:
        config = load_run_config(run_dir)
        requested = formats or config.export.formats
        model_dir = run_dir / "model"
        if not model_dir.is_dir() or not any(model_dir.iterdir()):
            raise ExportError(f"trained model not found in {model_dir}")
        exported_dir = run_dir / "exported"
        exported_dir.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, str] = {}

        is_adapter = (model_dir / "adapter_config.json").is_file()

        if ExportFormat.LORA_ADAPTER in requested:
            if not is_adapter:
                logger.warning("Run has no LoRA adapter (full fine-tune); skipping adapter export")
            else:
                adapter_out = exported_dir / "lora"
                if adapter_out.exists():
                    shutil.rmtree(adapter_out)
                shutil.copytree(model_dir, adapter_out)
                artifacts[ExportFormat.LORA_ADAPTER.value] = str(adapter_out)

        merged_dir = exported_dir / "merged"
        needs_merged = bool({ExportFormat.MERGED_SAFETENSORS, ExportFormat.GGUF} & set(requested))
        if needs_merged:
            if is_adapter:
                merge_lora_adapter(config, model_dir, merged_dir)
            else:
                if merged_dir.exists():
                    shutil.rmtree(merged_dir)
                shutil.copytree(model_dir, merged_dir)
            if ExportFormat.MERGED_SAFETENSORS in requested:
                artifacts[ExportFormat.MERGED_SAFETENSORS.value] = str(merged_dir)

        if ExportFormat.GGUF in requested:
            quantization = config.export.gguf_quantization
            gguf_path = exported_dir / f"model-{quantization.value}.gguf"
            convert_to_gguf(merged_dir, gguf_path, quantization)
            modelfile_path = write_modelfile(config.export.ollama, gguf_path)
            artifacts[ExportFormat.GGUF.value] = str(gguf_path)
            artifacts["modelfile"] = str(modelfile_path)
            logger.info(
                "Create the Ollama model with: ollama create {} -f {}",
                config.export.ollama.model_name,
                modelfile_path,
            )

        result = ExportResult(run_name=run_dir.name, artifacts=artifacts)
        (exported_dir / EXPORT_SUMMARY_FILENAME).write_text(
            json.dumps(
                {
                    "run_name": result.run_name,
                    "exported_at": datetime.now(UTC).isoformat(),
                    "artifacts": result.artifacts,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return result
