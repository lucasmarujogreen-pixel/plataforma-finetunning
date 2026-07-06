from pathlib import Path

from finetuning.core.hashing import sha256_dir, sha256_file, sha256_text


def test_sha256_text_is_deterministic() -> None:
    assert sha256_text("abc") == sha256_text("abc")
    assert sha256_text("abc") != sha256_text("abd")


def test_sha256_file(tmp_path: Path) -> None:
    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "b.txt"
    file_a.write_text("content")
    file_b.write_text("content")

    assert sha256_file(file_a) == sha256_file(file_b)


def test_sha256_dir_changes_with_content(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("one")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("two")
    first = sha256_dir(tmp_path)

    (tmp_path / "a.txt").write_text("changed")

    assert sha256_dir(tmp_path) != first


def test_sha256_dir_changes_with_renamed_file(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("one")
    first = sha256_dir(tmp_path)

    (tmp_path / "a.txt").rename(tmp_path / "renamed.txt")

    assert sha256_dir(tmp_path) != first
