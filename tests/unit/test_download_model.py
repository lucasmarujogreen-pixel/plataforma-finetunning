from pathlib import Path

from finetuning.application.download_model import DownloadModel
from finetuning.core.config.schemas import ModelConfig


class FakeModelStore:
    def __init__(self, snapshot_dir: Path, cached: bool) -> None:
        self._snapshot_dir = snapshot_dir
        self._cached = cached
        self.download_calls = 0

    def download(self, name: str, revision: str) -> Path:
        self.download_calls += 1
        return self._snapshot_dir

    def local_snapshot(self, name: str, revision: str) -> Path | None:
        return self._snapshot_dir if self._cached else None


def make_snapshot(tmp_path: Path) -> Path:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"\x00" * 2048)
    return snapshot


def test_download_always_syncs_snapshot(tmp_path: Path) -> None:
    store = FakeModelStore(make_snapshot(tmp_path), cached=False)
    result = DownloadModel(store).execute(ModelConfig(name="org/model"))

    assert store.download_calls == 1
    assert result.file_count == 2
    assert result.model_hash is not None


def test_download_called_even_when_partially_cached(tmp_path: Path) -> None:
    store = FakeModelStore(make_snapshot(tmp_path), cached=True)
    result = DownloadModel(store).execute(ModelConfig(name="org/model"))

    assert store.download_calls == 1
    assert result.path == store.local_snapshot("org/model", "main")


def test_hash_can_be_skipped(tmp_path: Path) -> None:
    store = FakeModelStore(make_snapshot(tmp_path), cached=True)
    result = DownloadModel(store).execute(ModelConfig(name="org/model"), compute_hash=False)

    assert result.model_hash is None
