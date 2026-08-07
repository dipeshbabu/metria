from __future__ import annotations

from pathlib import Path

import pytest

from metria import (
    ArtifactManifest,
    Capability,
    CapabilitySet,
    ComparisonPlan,
    HardwareFingerprint,
    ModelRef,
    RunRecord,
    RunSpec,
    RunStatus,
    RuntimeConfig,
    StudyRecipe,
    StudySpec,
    SupportLevel,
    WorkloadSpec,
    study_recipe_digest,
    study_recipe_to_data,
)
from metria.protocols import SupportReport


def _typed_run() -> RunSpec:
    return RunSpec(
        model=ModelRef(
            id="example/model",
            revision="abc123",
            tokenizer_id="example/tokenizer",
            tokenizer_revision="tok456",
            geometry={"head_dim": 128, "kv_heads": 8},
        ),
        runtime=RuntimeConfig(
            name="vllm",
            version="1.0",
            config={"dtype": "bfloat16", "kv_cache_dtype": "fp8"},
        ),
        scenario=WorkloadSpec(
            name="decode",
            config={"context": 4096, "max_tokens": 64},
        ),
        measurements=("decode_tps",),
        trial_policy={"warmup": 1, "repetitions": 3},
        environment_selector={"hardware_class": "h100"},
    )


def _mapping_run() -> RunSpec:
    return RunSpec(
        model={
            "id": "example/model",
            "revision": "abc123",
            "tokenizer_id": "example/tokenizer",
            "tokenizer_revision": "tok456",
            "geometry": {"head_dim": 128, "kv_heads": 8},
        },
        runtime={
            "name": "vllm",
            "version": "1.0",
            "dtype": "bfloat16",
            "kv_cache_dtype": "fp8",
        },
        scenario={"name": "decode", "context": 4096, "max_tokens": 64},
        measurements=("decode_tps",),
        trial_policy={"warmup": 1, "repetitions": 3},
        environment_selector={"hardware_class": "h100"},
    )


def _recipe(run: RunSpec) -> StudyRecipe:
    return StudyRecipe(
        study=StudySpec(
            name="typed-identity",
            runs=(run,),
            comparison=ComparisonPlan(control=frozenset({"model", "scenario"})),
        ),
        measurement_configs={"decode_tps": {"window": 16}},
        environment={"hardware_class": "h100"},
    )


def test_typed_run_normalizes_to_existing_mapping_contract() -> None:
    typed = _typed_run()
    mapping = _mapping_run()

    assert typed == mapping
    assert typed.model["id"] == "example/model"
    assert typed.runtime["kv_cache_dtype"] == "fp8"
    assert typed.scenario["context"] == 4096


def test_typed_recipe_preserves_v1_data_and_digest() -> None:
    typed = _recipe(_typed_run())
    mapping = _recipe(_mapping_run())

    assert study_recipe_to_data(typed) == study_recipe_to_data(mapping)
    assert study_recipe_digest(typed) == study_recipe_digest(mapping)


def test_identity_primitives_detach_nested_mutable_input() -> None:
    geometry = {"attention": {"kv_heads": [8]}}
    runtime_config = {"scheduler": {"modes": ["default"]}}
    workload_config = {"context_lengths": [4096, 8192]}

    model = ModelRef(id="example/model", geometry=geometry)
    runtime = RuntimeConfig(name="vllm", config=runtime_config)
    workload = WorkloadSpec(name="decode", config=workload_config)

    geometry["attention"]["kv_heads"].append(16)
    runtime_config["scheduler"]["modes"].append("mutated")
    workload_config["context_lengths"].append(16384)

    assert model["geometry"]["attention"]["kv_heads"] == (8,)
    assert runtime["scheduler"]["modes"] == ("default",)
    assert workload["context_lengths"] == (4096, 8192)


def test_model_runtime_and_workload_validate_public_identity() -> None:
    with pytest.raises(ValueError, match="at least an id or local path"):
        ModelRef()
    with pytest.raises(ValueError, match="reserved fields"):
        RuntimeConfig(name="vllm", config={"name": "other"})
    with pytest.raises(ValueError, match="name or configuration"):
        WorkloadSpec()


def test_model_and_artifact_paths_normalize_pathlike_values(tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    artifact_path = tmp_path / "result.json"

    model = ModelRef(path=model_path)
    artifact = ArtifactManifest(name="result", kind="report", path=artifact_path)

    assert model["path"] == str(model_path)
    assert artifact["path"] == str(artifact_path)


def test_capability_set_uses_conservative_support_vocabulary() -> None:
    supported = Capability(
        name="kv.fp8",
        status="supported",
        reasons=("runtime reports native support",),
        evidence={"runtime": {"version": "1"}},
    )
    unknown = Capability(name="attention.backend", status=SupportLevel.UNKNOWN)
    capabilities = CapabilitySet((supported, unknown))

    assert supported.status is SupportLevel.SUPPORTED
    assert capabilities.get("kv.fp8") is supported
    assert capabilities.to_mapping()["attention.backend"]["status"] == "unknown"

    with pytest.raises(ValueError, match="unique"):
        CapabilitySet((supported, supported))
    with pytest.raises(ValueError, match="must be one of"):
        Capability(name="bad", status="maybe")


def test_support_report_normalizes_to_shared_support_level() -> None:
    report = SupportReport(
        status="experimental",
        reasons=("requires patched runtime",),
        evidence={"patch": "abc"},
    )

    assert report.status is SupportLevel.EXPERIMENTAL
    assert report.status == "experimental"

    with pytest.raises(ValueError, match="support status must be one of"):
        SupportReport(status="partial")


def test_hardware_fingerprint_is_observed_mapping_evidence() -> None:
    accelerators = [{"name": "H100", "memory_bytes": 80_000_000_000}]
    fingerprint = HardwareFingerprint(
        platform={"system": "Linux", "arch": "x86_64"},
        host={"cpu_count": 32},
        accelerators=tuple(accelerators),
        software={"cuda": "13.0", "driver": "999.0"},
    )
    accelerators[0]["name"] = "mutated"

    assert fingerprint["accelerators"][0]["name"] == "H100"
    assert fingerprint["software"]["cuda"] == "13.0"


def test_artifact_manifest_validates_identity_and_runs_as_record_artifact() -> None:
    source = {"repository": "https://example.invalid/repo", "refs": ["abc"]}
    manifest = ArtifactManifest(
        name="model",
        kind="model",
        uri="hf://example/model",
        revision="abc123",
        sha256="A" * 64,
        size_bytes=123,
        source=source,
    )
    source["refs"].append("mutated")

    record = RunRecord(
        study_name="artifact-study",
        run_id="one",
        requested=_mapping_run(),
        resolved={},
        observed={},
        status=RunStatus.COMPLETED,
        artifacts=(manifest,),
    )

    assert manifest.sha256 == "a" * 64
    assert manifest["source"]["refs"] == ("abc",)
    assert record.artifacts[0]["name"] == "model"
    assert record.artifacts[0]["sha256"] == "a" * 64

    with pytest.raises(ValueError, match="64 hexadecimal"):
        ArtifactManifest(name="bad", kind="model", sha256="abc")
    with pytest.raises(ValueError, match="non-negative"):
        ArtifactManifest(name="bad", kind="model", uri="x", size_bytes=-1)
    with pytest.raises(ValueError, match="uri, path, or sha256"):
        ArtifactManifest(name="bad", kind="model")
