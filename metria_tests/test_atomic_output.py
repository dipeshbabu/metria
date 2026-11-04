"""Publishing a new evidence file must never replace another directory entry."""

from __future__ import annotations

import os

import pytest

from metria.verification import _write_atomic


def test_new_output_is_complete_and_temporary_file_is_removed(tmp_path):
    output = tmp_path / "result.json"
    _write_atomic(output, '{"complete": true}\n')
    assert output.read_text() == '{"complete": true}\n'
    assert not (tmp_path / ".result.json.tmp").exists()


def test_existing_output_is_preserved(tmp_path):
    output = tmp_path / "result.json"
    output.write_text("existing")
    with pytest.raises(FileExistsError):
        _write_atomic(output, "replacement")
    assert output.read_text() == "existing"
    assert not (tmp_path / ".result.json.tmp").exists()


def test_competing_publisher_cannot_be_overwritten(tmp_path, monkeypatch):
    output = tmp_path / "result.json"
    operation = "rename" if os.name == "nt" else "link"
    original_publish = getattr(os, operation)

    def competing_publish(source, destination):
        output.write_text("other writer")
        return original_publish(source, destination)

    monkeypatch.setattr(os, operation, competing_publish)
    with pytest.raises(FileExistsError):
        _write_atomic(output, "replacement")
    assert output.read_text() == "other writer"
    assert not (tmp_path / ".result.json.tmp").exists()


def test_dangling_output_symlink_is_preserved(tmp_path):
    output = tmp_path / "result.json"
    target = tmp_path / "missing.json"
    try:
        output.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable to this Windows account")
    with pytest.raises(FileExistsError):
        _write_atomic(output, "replacement")
    assert output.is_symlink()
    assert not target.exists()


def test_another_writers_temporary_file_is_not_deleted(tmp_path):
    temporary = tmp_path / ".result.json.tmp"
    temporary.write_text("other writer")
    with pytest.raises(FileExistsError):
        _write_atomic(tmp_path / "result.json", "replacement")
    assert temporary.read_text() == "other writer"
