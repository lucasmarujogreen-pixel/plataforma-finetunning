from finetuning.infrastructure.environment import collect_git_commit, collect_library_versions


def test_collect_library_versions_includes_core_libraries() -> None:
    versions = collect_library_versions()

    assert "python" in versions
    assert versions["torch"] != "not-installed"
    assert versions["transformers"] != "not-installed"
    assert versions["trl"] != "not-installed"


def test_collect_git_commit_returns_hash_or_none() -> None:
    commit = collect_git_commit()

    assert commit is None or len(commit) == 40
