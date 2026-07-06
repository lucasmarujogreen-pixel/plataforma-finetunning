import json
from pathlib import Path

import pytest

from finetuning.application.compare_experiments import CompareExperiments, summarize_run
from finetuning.core.config import AppConfig, load_config
from finetuning.core.enums import DatasetSchema
from finetuning.core.exceptions import ExperimentError
from finetuning.domain.entities import DatasetMetadata
from finetuning.infrastructure.experiment_manager import ExperimentManager
from finetuning.monitoring.hardware import HardwareProfile


def make_run(
    runs_dir: Path,
    config: AppConfig,
    eval_loss: float,
    vram: float | None = 4000.0,
) -> Path:
    manager = ExperimentManager(runs_dir)
    metadata = DatasetMetadata(
        name="example",
        version="v1",
        source_path="x.jsonl",
        source_sha256="deadbeef",
        record_schema=DatasetSchema.CHAT,
        num_records=10,
        num_tokens=100,
        mean_tokens=10.0,
        max_tokens=20,
        language="pt",
        created_at="2026-07-02T00:00:00+00:00",
    )
    profile = HardwareProfile(cuda_available=False, cuda_version=None, driver_version=None)
    run = manager.create_run(config, metadata, profile)
    manager.finalize_run(
        run,
        status="completed",
        total_seconds=60.0,
        final_metrics={"train_loss": eval_loss + 0.1, "eval_loss": eval_loss},
    )
    if vram is not None:
        metrics_rows = [
            {"step": 5, "loss": 2.0, "sys_vram_used_mb": vram - 500},
            {"step": 10, "loss": 1.5, "sys_vram_used_mb": vram},
        ]
        (run.metrics_dir / "training_log.jsonl").write_text(
            "\n".join(json.dumps(row) for row in metrics_rows), encoding="utf-8"
        )
    return run.root


def test_summarize_run_extracts_config_and_metrics(config_dir: Path, tmp_path: Path) -> None:
    config = load_config(config_dir)
    run_dir = make_run(tmp_path / "runs", config, eval_loss=1.5)

    summary = summarize_run(run_dir)

    assert summary.status == "completed"
    assert summary.method == "qlora"
    assert summary.eval_loss == 1.5
    assert summary.lora_r == config.lora.r
    assert summary.peak_vram_mb == 4000.0
    assert summary.effective_batch_size == (
        config.training.micro_batch_size * config.training.gradient_accumulation_steps
    )


def test_compare_writes_markdown_report(config_dir: Path, tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    config_a = load_config(config_dir)
    config_b = load_config(config_dir, overrides=["lora.r=8", "optimizer.learning_rate=1e-4"])
    make_run(runs_dir, config_a, eval_loss=1.5)
    make_run(runs_dir, config_b, eval_loss=1.2, vram=None)

    comparator = CompareExperiments(runs_dir)
    summaries = comparator.execute()
    report_path = comparator.write_report(summaries)

    assert len(summaries) == 2
    content = report_path.read_text(encoding="utf-8")
    assert "Best eval_loss" in content
    assert "1.2" in content


def test_compare_rejects_unknown_run(config_dir: Path, tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    make_run(runs_dir, load_config(config_dir), eval_loss=1.5)

    with pytest.raises(ExperimentError, match="not found"):
        CompareExperiments(runs_dir).execute(["does-not-exist"])


def test_compare_empty_runs_dir(tmp_path: Path) -> None:
    with pytest.raises(ExperimentError, match="no experiment runs"):
        CompareExperiments(tmp_path / "runs").execute()
