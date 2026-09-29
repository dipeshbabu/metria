from __future__ import annotations

import pytest

from metria.runtimes.vllm_capability import fp8_cache_gap


@pytest.mark.parametrize("dtype", ["fp8", "fp8_e4m3", "fp8_e5m2"])
def test_confirmed_turing_fp8_failure_is_actionable_before_engine_launch(dtype):
    reason = fp8_cache_gap(
        "0.30.0+cu129",
        dtype,
        {"status": "observed", "device_type": "cuda", "compute_capability": [7, 5]},
    )
    assert (
        reason is not None and "SM89+" in reason and "automatic cache dtype" in reason
    )


@pytest.mark.parametrize(
    "version,dtype,hardware",
    [
        (
            "0.30.0+cu129",
            "auto",
            {"status": "observed", "device_type": "cuda", "compute_capability": [7, 5]},
        ),
        (
            "0.31.0+cu129",
            "fp8",
            {"status": "observed", "device_type": "cuda", "compute_capability": [7, 5]},
        ),
        ("0.30.0+cu129", "fp8", {"status": "unknown"}),
        ("0.30.0+cu129", "fp8", {"status": "observed", "device_type": "cpu"}),
        (
            "0.30.0+cu129",
            "fp8",
            {"status": "observed", "device_type": "cuda", "compute_capability": [8, 9]},
        ),
        (
            "0.30.0+cu129",
            "fp8",
            {
                "status": "observed",
                "device_type": "cuda",
                "compute_capability": [True, 5],
            },
        ),
        (
            "0.30.0+cu129",
            "fp8",
            {"status": "observed", "device_type": "cuda", "compute_capability": None},
        ),
    ],
)
def test_guard_does_not_invent_support_or_reject_unqualified_combinations(
    version, dtype, hardware
):
    assert fp8_cache_gap(version, dtype, hardware) is None


def test_adapter_rejects_confirmed_fp8_hardware_without_constructing_engine(
    monkeypatch,
):
    from metria import RunSpec, TreatmentSpec, TreatmentType
    from metria.runtimes import vllm

    monkeypatch.setattr(vllm, "_vllm_available", lambda: True)
    monkeypatch.setattr(vllm, "_vllm_version", lambda: "0.30.0+cu129")
    monkeypatch.setattr(
        vllm,
        "native_hardware",
        lambda: {
            "status": "observed",
            "device_type": "cuda",
            "compute_capability": [7, 5],
        },
    )

    def no_engine():
        raise AssertionError("preflight must not construct an engine")

    monkeypatch.setattr(vllm, "_load_vllm", no_engine)
    spec = RunSpec(
        model={"id": "fixture/model"},
        runtime={"name": "vllm", "version": "0.30.0+cu129"},
        scenario={},
        measurements=(),
        treatments=(
            TreatmentSpec(
                name="vllm.kv_cache",
                kind=TreatmentType.RUNTIME_FEATURE,
                config={"dtype": "fp8"},
            ),
        ),
    )
    result = vllm.VLLMAdapter().probe(spec, {})
    assert result.status == "unsupported"
    assert any("SM89+" in reason for reason in result.reasons)
    assert result.evidence["unsupported_cache_hardware"]["compute_capability"] == (7, 5)


def test_retained_native_failure_and_installed_preflight_are_consistent():
    import hashlib
    import json
    from pathlib import Path

    root = Path(__file__).parents[1] / "artifacts/qualification/fp8-unsupported"
    hashes = json.loads((root / "sha256.json").read_text())
    for name, digest in hashes.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    evidence = json.loads((root / "preflight.json").read_text())
    assert evidence["positive_qualification"] is False
    assert evidence["preflight"]["status"] == "unsupported"
    assert evidence["hardware"]["compute_capability"] == [7, 5]
    assert "FP8" in (root / "native-stderr.log").read_text()
