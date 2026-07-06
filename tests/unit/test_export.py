from pathlib import Path

import pytest

from finetuning.application.export_model import ExportModel
from finetuning.core.config.schemas import OllamaConfig
from finetuning.core.enums import GGUFQuantization
from finetuning.core.exceptions import ExportError
from finetuning.exporters.gguf import convert_to_gguf, find_convert_script, find_quantize_binary
from finetuning.exporters.merge import merge_lora_adapter
from finetuning.exporters.modelfile import build_modelfile, write_modelfile


class TestModelfile:
    def test_build_contains_parameters(self) -> None:
        ollama = OllamaConfig(model_name="meu-modelo", temperature=0.5, top_p=0.8, num_ctx=2048)

        content = build_modelfile(ollama, "model-q4_k_m.gguf")

        assert "FROM ./model-q4_k_m.gguf" in content
        assert "PARAMETER temperature 0.5" in content
        assert "PARAMETER top_p 0.8" in content
        assert "PARAMETER num_ctx 2048" in content
        assert "SYSTEM" not in content

    def test_build_includes_system_prompt(self) -> None:
        ollama = OllamaConfig(system_prompt="Você é um assistente de análises.")

        content = build_modelfile(ollama, "model.gguf")

        assert 'SYSTEM """Você é um assistente de análises."""' in content

    def test_write_modelfile(self, tmp_path: Path) -> None:
        gguf_path = tmp_path / "model.gguf"
        gguf_path.write_bytes(b"fake")

        modelfile_path = write_modelfile(OllamaConfig(), gguf_path)

        assert modelfile_path.name == "Modelfile"
        assert "FROM ./model.gguf" in modelfile_path.read_text(encoding="utf-8")


class TestGGUF:
    def test_find_convert_script_missing(self, tmp_path: Path) -> None:
        assert find_convert_script(tmp_path) is None

    def test_find_quantize_binary_missing(self, tmp_path: Path) -> None:
        assert find_quantize_binary(tmp_path) is None

    def test_convert_requires_llamacpp(self, tmp_path: Path) -> None:
        with pytest.raises(ExportError, match="setup_llamacpp"):
            convert_to_gguf(
                tmp_path,
                tmp_path / "model.gguf",
                GGUFQuantization.Q4_K_M,
                llamacpp_dir=tmp_path / "llama.cpp",
            )


class TestExportModel:
    def test_rejects_run_without_model(self, tmp_path: Path, config_dir: Path) -> None:
        import yaml

        from finetuning.core.config import load_config

        run_dir = tmp_path / "run"
        (run_dir / "model").mkdir(parents=True)
        (run_dir / "configs").mkdir()
        config = load_config(config_dir)
        (run_dir / "configs" / "resolved.yaml").write_text(
            yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8"
        )

        with pytest.raises(ExportError, match="trained model not found"):
            ExportModel().execute(run_dir)


class TestMerge:
    def test_rejects_missing_adapter(self, tmp_path: Path, config_dir: Path) -> None:
        from finetuning.core.config import load_config

        config = load_config(config_dir)

        with pytest.raises(ExportError, match="no LoRA adapter"):
            merge_lora_adapter(config, tmp_path, tmp_path / "out")
