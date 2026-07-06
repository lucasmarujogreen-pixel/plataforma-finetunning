from pathlib import Path

import pytest
import yaml

from finetuning.core.config import AppConfig, load_config
from finetuning.core.enums import DatasetSchema
from finetuning.core.exceptions import ExperimentError
from finetuning.domain.entities import DatasetMetadata
from finetuning.infrastructure.experiment_manager import ExperimentManager
from finetuning.monitoring.hardware import HardwareProfile


@pytest.fixture()
def app_config(config_dir: Path) -> AppConfig:
    return load_config(config_dir)


def make_dataset_metadata() -> DatasetMetadata:
    return DatasetMetadata(
        name="example",
        version="abc123",
        source_path="datasets/raw/example.jsonl",
        source_sha256="deadbeef",
        record_schema=DatasetSchema.CHAT,
        num_records=10,
        num_tokens=100,
        mean_tokens=10.0,
        max_tokens=20,
        language="pt",
        created_at="2026-07-02T00:00:00+00:00",
        splits={"train": 9, "validation": 1},
    )


def make_profile() -> HardwareProfile:
    return HardwareProfile(cuda_available=False, cuda_version=None, driver_version=None)


def test_create_run_builds_structure_and_manifest(tmp_path: Path, app_config: AppConfig) -> None:
    manager = ExperimentManager(tmp_path / "runs")

    run = manager.create_run(app_config, make_dataset_metadata(), make_profile())

    for subdirectory in ("configs", "checkpoints", "logs", "metrics", "plots", "exported"):
        assert (run.root / subdirectory).is_dir()
    manifest = ExperimentManager.load_manifest(run.root)
    assert manifest["status"] == "running"
    assert manifest["model_name"] == app_config.model.name
    assert manifest["seed"] == app_config.training.seed
    assert manifest["dataset"]["num_records"] == 10
    assert manifest["libraries"]["torch"] != "not-installed"
    assert len(manifest["config_hash"]) == 64


def test_resolved_config_round_trips(tmp_path: Path, app_config: AppConfig) -> None:
    manager = ExperimentManager(tmp_path / "runs")
    run = manager.create_run(app_config, make_dataset_metadata(), make_profile())

    raw = yaml.safe_load(run.resolved_config_path.read_text(encoding="utf-8"))
    restored = AppConfig.model_validate(raw)

    assert restored == app_config


def test_finalize_run_updates_manifest(tmp_path: Path, app_config: AppConfig) -> None:
    manager = ExperimentManager(tmp_path / "runs")
    run = manager.create_run(app_config, make_dataset_metadata(), make_profile())

    manager.finalize_run(run, status="completed", total_seconds=12.5, final_metrics={"loss": 1.0})

    manifest = ExperimentManager.load_manifest(run.root)
    assert manifest["status"] == "completed"
    assert manifest["total_seconds"] == 12.5
    assert manifest["final_metrics"] == {"loss": 1.0}


def test_list_runs_returns_manifests(tmp_path: Path, app_config: AppConfig) -> None:
    manager = ExperimentManager(tmp_path / "runs")
    manager.create_run(app_config, make_dataset_metadata(), make_profile())

    assert len(manager.list_runs()) == 1


def test_load_manifest_missing(tmp_path: Path) -> None:
    with pytest.raises(ExperimentError):
        ExperimentManager.load_manifest(tmp_path)
