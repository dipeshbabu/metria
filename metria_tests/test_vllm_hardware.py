import hashlib
from types import SimpleNamespace

import pytest

from metria.runtimes import vllm_hardware as hardware


def test_native_cpu_identity_is_not_guessed_from_environment(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        hardware,
        "importlib",
        SimpleNamespace(
            import_module=lambda _: SimpleNamespace(
                current_platform=SimpleNamespace(device_type="cpu")
            )
        ),
    )
    assert hardware.native_hardware() == {
        "status": "observed",
        "device_type": "cpu",
        "source": "vllm_native_platform",
    }


def test_gpu_identity_uses_device_properties_and_hashes_uuid(monkeypatch):
    platform = SimpleNamespace(current_platform=SimpleNamespace(device_type="cuda"))
    properties = SimpleNamespace(
        name="GPU fixture",
        major=7,
        minor=5,
        total_memory=4096,
        uuid="private-device-uuid",
    )
    torch = SimpleNamespace(
        version=SimpleNamespace(cuda="12.9", hip=None),
        cuda=SimpleNamespace(
            current_device=lambda: 0, get_device_properties=lambda _: properties
        ),
    )
    monkeypatch.setattr(
        hardware,
        "importlib",
        SimpleNamespace(
            import_module=lambda name: platform if name == "vllm.platforms" else torch
        ),
    )
    result = hardware.native_hardware()
    assert result["device_type"] == "cuda" and result["capacity_bytes"] == 4096
    assert result["uuid_sha256"] == hashlib.sha256(properties.uuid.encode()).hexdigest()
    assert properties.uuid not in str(result)
    torch.version.hip = "7.2"
    assert hardware.native_hardware()["device_type"] == "rocm"


@pytest.mark.parametrize("kind", ["unknown", "exception"])
def test_unobservable_platform_stays_unknown(monkeypatch, kind):
    def load(name):
        if kind == "exception":
            raise ImportError("unavailable")
        return SimpleNamespace(current_platform=SimpleNamespace(device_type="tpu"))

    monkeypatch.setattr(hardware, "importlib", SimpleNamespace(import_module=load))
    assert hardware.native_hardware()["status"] == "unknown"
