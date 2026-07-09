"""Experiment run creation, manifest persistence and lookup."""

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml
from loguru import logger

from finetuning.core.config.reranker_schemas import RerankerAppConfig
from finetuning.core.config.schemas import AppConfig
from finetuning.core.exceptions import ExperimentError
from finetuning.core.hashing import sha256_text
from finetuning.domain.entities import DatasetMetadata
from finetuning.infrastructure.environment import collect_git_commit, collect_library_versions
from finetuning.monitoring.hardware import HardwareProfile

MANIFEST_FILENAME = "manifest.json"
RESOLVED_CONFIG_FILENAME = "resolved.yaml"
RUN_SUBDIRECTORIES = (
    "configs",
    "checkpoints",
    "logs",
    "metrics",
    "plots",
    "exported",
    "evaluation",
    "model",
)


@dataclass(frozen=True)
class ExperimentRun:
    experiment_id: str
    name: str
    root: Path

    @property
    def configs_dir(self) -> Path:
        return self.root / "configs"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def metrics_dir(self) -> Path:
        return self.root / "metrics"

    @property
    def plots_dir(self) -> Path:
        return self.root / "plots"

    @property
    def exported_dir(self) -> Path:
        return self.root / "exported"

    @property
    def evaluation_dir(self) -> Path:
        return self.root / "evaluation"

    @property
    def model_dir(self) -> Path:
        return self.root / "model"

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_FILENAME

    @property
    def resolved_config_path(self) -> Path:
        return self.configs_dir / RESOLVED_CONFIG_FILENAME


class ExperimentManager:
    def __init__(self, runs_dir: Path) -> None:
        self._runs_dir = runs_dir

    def create_run(
        self,
        config: AppConfig | RerankerAppConfig,
        dataset_metadata: DatasetMetadata,
        hardware_profile: HardwareProfile,
        extras: dict[str, Any] | None = None,
    ) -> ExperimentRun:
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
        model_slug = config.model.name.split("/")[-1].lower()
        name = f"{timestamp}_{model_slug}"
        if config.experiment.name:
            name = f"{name}_{config.experiment.name}"
        experiment_id = str(uuid.uuid4())
        if (self._runs_dir / name).exists():
            name = f"{name}_{experiment_id[:8]}"
        run = ExperimentRun(experiment_id=experiment_id, name=name, root=self._runs_dir / name)
        for subdirectory in RUN_SUBDIRECTORIES:
            (run.root / subdirectory).mkdir(parents=True)

        resolved = config.model_dump(mode="json")
        run.resolved_config_path.write_text(
            yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        manifest: dict[str, Any] = {
            "experiment_id": run.experiment_id,
            "name": run.name,
            "created_at": datetime.now(UTC).isoformat(),
            "status": "running",
            "model_name": config.model.name,
            "model_revision": config.model.revision,
            "training_method": config.training.method.value,
            "seed": config.training.seed,
            "config_hash": sha256_text(config.model_dump_json()),
            "dataset": asdict(dataset_metadata),
            "git_commit": collect_git_commit(),
            "libraries": collect_library_versions(),
            "hardware": asdict(hardware_profile),
            "tags": dict(config.experiment.tags),
            "total_seconds": None,
            "final_metrics": None,
        }
        manifest.update(extras or {})
        run.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        logger.info("Created experiment run {} ({})", run.name, run.experiment_id)
        return run

    def finalize_run(
        self,
        run: ExperimentRun,
        status: str,
        total_seconds: float,
        final_metrics: dict[str, float] | None,
    ) -> None:
        manifest = self.load_manifest(run.root)
        manifest["status"] = status
        manifest["total_seconds"] = round(total_seconds, 2)
        manifest["final_metrics"] = final_metrics
        run.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        logger.info("Run {} finalized with status '{}'", run.name, status)

    def list_runs(self) -> list[dict[str, Any]]:
        if not self._runs_dir.is_dir():
            return []
        manifests = []
        for run_dir in sorted(self._runs_dir.iterdir()):
            manifest_path = run_dir / MANIFEST_FILENAME
            if manifest_path.is_file():
                manifests.append(self.load_manifest(run_dir))
        return manifests

    @staticmethod
    def load_manifest(run_dir: Path) -> dict[str, Any]:
        manifest_path = run_dir / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise ExperimentError(f"manifest not found in {run_dir}")
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    @staticmethod
    def load_run(run_dir: Path) -> ExperimentRun:
        manifest = ExperimentManager.load_manifest(run_dir)
        return ExperimentRun(
            experiment_id=manifest["experiment_id"],
            name=manifest["name"],
            root=run_dir,
        )


def _load_resolved_config(run_dir: Path) -> dict[str, Any]:
    resolved_path = run_dir / "configs" / RESOLVED_CONFIG_FILENAME
    if not resolved_path.is_file():
        raise ExperimentError(f"resolved config not found at {resolved_path}")
    raw = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    return cast(dict[str, Any], raw)


def load_run_config(run_dir: Path) -> AppConfig:
    return AppConfig.model_validate(_load_resolved_config(run_dir))


def load_reranker_run_config(run_dir: Path) -> RerankerAppConfig:
    return RerankerAppConfig.model_validate(_load_resolved_config(run_dir))
