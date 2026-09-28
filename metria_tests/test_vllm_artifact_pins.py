import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from metria.runtimes import vllm_artifacts as artifacts


@pytest.fixture
def model(tmp_path):
    names = (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "model.safetensors",
    )
    pins = {}
    for name in names:
        content = ("fixture " + name).encode()
        (tmp_path / name).write_bytes(content)
        pins[name] = hashlib.sha256(content).hexdigest()
    return tmp_path, pins


def test_pins_identify_content_independently_of_root_path(model, tmp_path):
    directory, pins = model
    first = artifacts.verify_model_files(directory, pins)
    assert first["status"] == "verified" and len(first["files"]) == 4
    assert str(directory) not in str(first)
    (directory / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="content"):
        artifacts.verify_model_files(directory, pins)


@pytest.mark.parametrize(
    "name",
    [
        "../model.bin",
        "/model.bin",
        "C:/model.bin",
        "sub\\model.bin",
        "sub/../model.bin",
        "model.py",
    ],
)
def test_manifest_rejects_unsafe_or_untracked_payload_paths(model, name):
    _, pins = model
    with pytest.raises(ValueError):
        artifacts.validate_file_pins({**pins, name: "a" * 64})


def test_extra_payload_cannot_silently_override_pinned_weights(model):
    directory, pins = model
    (directory / "pytorch_model.bin").write_bytes(b"untracked alternative")
    with pytest.raises(ValueError, match="inventory"):
        artifacts.verify_model_files(directory, pins)


def test_missing_tokenizer_pin_fails_before_content_verification(model):
    directory, pins = model
    del pins["tokenizer.json"]
    with pytest.raises(ValueError, match="tokenizer"):
        artifacts.verify_model_files(directory, pins)


def test_runtime_fingerprint_hashes_actual_installed_payload(tmp_path, monkeypatch):
    package = tmp_path / "vllm"
    package.mkdir()
    init = package / "__init__.py"
    init.write_text("__version__ = 'qualified'")
    distribution = SimpleNamespace(
        version="qualified",
        files=[Path("vllm/__init__.py")],
        locate_file=lambda path: tmp_path / path,
    )
    monkeypatch.setattr(
        artifacts.importlib.metadata, "distribution", lambda name: distribution
    )
    monkeypatch.setattr(artifacts.importlib.metadata, "version", lambda name: "pinned")
    before = artifacts.installed_runtime_identity(
        loaded_module=SimpleNamespace(__file__=str(init))
    )
    assert (
        artifacts.require_runtime_pin({"vllm_distribution_sha256": before["sha256"]})
        == before
    )
    assert artifacts.require_runtime_pin({}) is None
    init.write_text("modified runtime")
    after = artifacts.installed_runtime_identity()
    assert before["sha256"] != after["sha256"]
    with pytest.raises(ValueError, match="differ"):
        artifacts.require_runtime_pin({"vllm_distribution_sha256": before["sha256"]})
    with pytest.raises(ValueError, match="loaded"):
        artifacts.installed_runtime_identity(
            loaded_module=SimpleNamespace(__file__=str(tmp_path / "shadow.py"))
        )


@pytest.mark.parametrize("value", [True, "", "A" * 64, "a" * 63])
def test_runtime_pin_format_is_validated_before_inspection(value):
    with pytest.raises(ValueError, match="SHA256"):
        artifacts.require_runtime_pin({"vllm_distribution_sha256": value})
