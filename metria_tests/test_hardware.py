from __future__ import annotations

import re

from metria.hardware import capture_hardware_fingerprint


def test_hardware_fingerprint_is_portable_and_privacy_conscious() -> None:
    fingerprint = capture_hardware_fingerprint(metadata={"purpose": "test"})
    data = fingerprint.to_mapping()

    assert data["platform"]["system"]
    assert data["platform"]["machine"] is not None
    assert data["host"]["cpu_count"] is None or data["host"]["cpu_count"] > 0
    assert re.fullmatch(r"[0-9a-f]{64}", data["host"]["hostname_sha256"])
    assert "hostname" not in data["host"]
    assert data["software"]["python_version"]
    assert data["metadata"]["accelerator_detection"] == "runtime_or_adapter_required"
    assert data["metadata"]["purpose"] == "test"
