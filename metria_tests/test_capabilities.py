from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from metria import RunSpec, RunStatus, TreatmentSpec, TreatmentType, execute_run
from metria.capabilities import (
    ModelGeometry,
    evaluate_turboquant_kv_capability,
    inspect_model_geometry,
)
from metria.identity import SupportLevel
from metria.inspection import inspect_run_capabilities
from metria.protocols import (
    CaptureRequest,
    InferenceBatch,
    InferenceRequest,
    MeasurementResult,
    SupportReport,
)


def _turbo_spec(
    *,
    head_dim: int | None,
    override: bool = False,
) -> RunSpec:
    geometry = {} if head_dim is None else {"head_dim": head_dim}
    trial_policy: dict[str, Any] = {}
    if override:
        trial_policy["capability_overrides"] = ("turboquant.kv_cache.geometry",)
    return RunSpec(
        model={"id": "example/model", "geometry": geometry},
        runtime={"name": "llamacpp"},
        scenario={"max_tokens": 4},
        measurements=("test.measurement",),
        treatments=(
            TreatmentSpec(
                name="llamacpp.kv_cache",
                kind=TreatmentType.RUNTIME_FEATURE,
                config={"key_dtype": "q8_0", "value_dtype": "turbo3"},
            ),
        ),
        trial_policy=trial_policy,
    )


def test_model_geometry_derives_head_dim_from_consistent_metadata() -> None:
    inspection = inspect_model_geometry(
        {
            "geometry": {
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "num_hidden_layers": 32,
                "max_position_embeddings": 8192,
            }
        }
    )

    assert inspection.capability.status is SupportLevel.SUPPORTED
    assert inspection.geometry == ModelGeometry(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        num_hidden_layers=32,
        context_length=8192,
        evidence=inspection.geometry.evidence if inspection.geometry else {},
    )
    assert inspection.geometry is not None
    assert inspection.geometry.head_dim == 128


def test_model_geometry_conflict_fails_to_unknown() -> None:
    inspection = inspect_model_geometry(
        {
            "geometry": {
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "head_dim": 64,
            }
        }
    )

    assert inspection.geometry is None
    assert inspection.capability.status is SupportLevel.UNKNOWN
    assert "contradicts" in inspection.capability.reasons[0]


def test_turboquant_validated_head_dims_are_supported() -> None:
    for head_dim in (128, 256):
        capability = evaluate_turboquant_kv_capability(
            {"geometry": {"head_dim": head_dim}},
            {"key_dtype": "q8_0", "value_dtype": "turbo3"},
        )
        assert capability.status is SupportLevel.SUPPORTED
        assert capability.evidence["head_dim"] == head_dim


def test_turboquant_head_dim_64_fails_closed_without_override() -> None:
    result = inspect_run_capabilities(_turbo_spec(head_dim=64))
    capability = result.capabilities.get("turboquant.kv_cache.geometry")

    assert capability is not None
    assert capability.status is SupportLevel.UNSUPPORTED
    assert result.blocking == (capability,)


def test_turboquant_head_dim_64_explicit_override_is_experimental_and_allowed() -> None:
    result = inspect_run_capabilities(_turbo_spec(head_dim=64, override=True))
    capability = result.capabilities.get("turboquant.kv_cache.geometry")

    assert capability is not None
    assert capability.status is SupportLevel.EXPERIMENTAL
    assert capability.evidence["experimental_override"] is True
    assert result.blocking == ()


def test_unvalidated_consistent_head_dim_requires_explicit_override() -> None:
    blocked = inspect_run_capabilities(_turbo_spec(head_dim=96))
    allowed = inspect_run_capabilities(_turbo_spec(head_dim=96, override=True))

    assert (
        blocked.capabilities.get("turboquant.kv_cache.geometry").status
        is SupportLevel.EXPERIMENTAL
    )
    assert blocked.blocking
    assert allowed.blocking == ()


def test_missing_turboquant_geometry_remains_unknown_even_with_override() -> None:
    result = inspect_run_capabilities(_turbo_spec(head_dim=None, override=True))
    capability = result.capabilities.get("turboquant.kv_cache.geometry")

    assert capability is not None
    assert capability.status is SupportLevel.UNKNOWN
    assert result.blocking == (capability,)


def test_non_turbo_kv_configuration_does_not_require_geometry() -> None:
    capability = evaluate_turboquant_kv_capability(
        {},
        {"key_dtype": "q8_0", "value_dtype": "q8_0"},
    )

    assert capability.status is SupportLevel.SUPPORTED
    assert capability.evidence["active"] is False


class _CountingSession:
    def infer(
        self,
        requests: Sequence[InferenceRequest],
        capture: Sequence[CaptureRequest] = (),
    ) -> InferenceBatch:
        del capture
        return InferenceBatch(outputs=tuple("ok" for _ in requests))

    def reset(self, scope: str = "measurement") -> None:
        del scope

    def close(self) -> None:
        pass


class _CountingAdapter:
    name = "llamacpp"

    def __init__(self) -> None:
        self.probes = 0

    def probe(
        self,
        spec: RunSpec,
        environment: Mapping[str, Any],
    ) -> SupportReport:
        del spec, environment
        self.probes += 1
        return SupportReport(status="supported")

    def resolve(
        self,
        spec: RunSpec,
        environment: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del spec, environment
        return {"runtime": {"name": self.name}}

    def launch(
        self,
        resolved: Mapping[str, Any],
        environment: Mapping[str, Any],
    ) -> _CountingSession:
        del resolved, environment
        return _CountingSession()

    def observe(self, session: _CountingSession) -> Mapping[str, Any]:
        del session
        return {"runtime": self.name}


class _Measurement:
    name = "test.measurement"
    version = "1"

    def requirements(self, config: Mapping[str, Any]) -> tuple[CaptureRequest, ...]:
        del config
        return ()

    def execute(
        self,
        session: _CountingSession,
        scenario: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> MeasurementResult:
        del session, scenario, config
        return MeasurementResult()


def test_execute_run_blocks_known_unsupported_capability_before_adapter_probe() -> None:
    adapter = _CountingAdapter()

    record = execute_run(
        study_name="guardrail-study",
        run_id="run-0",
        spec=_turbo_spec(head_dim=64),
        adapter=adapter,
        measurement=_Measurement(),
        measurement_config={},
        environment={},
    )

    assert record.status is RunStatus.PREFLIGHT_FAILED
    assert adapter.probes == 0
    assert record.events[-1]["kind"] == "capability_blocked"
    assert record.provenance["capabilities"]["allowed"] is False


def test_execute_run_retains_experimental_override_in_provenance() -> None:
    adapter = _CountingAdapter()

    record = execute_run(
        study_name="guardrail-study",
        run_id="run-0",
        spec=_turbo_spec(head_dim=64, override=True),
        adapter=adapter,
        measurement=_Measurement(),
        measurement_config={},
        environment={},
    )

    assert record.status is RunStatus.COMPLETED
    assert adapter.probes == 1
    capability = record.provenance["capabilities"]["capabilities"][
        "turboquant.kv_cache.geometry"
    ]
    assert capability["status"] == "experimental"
    assert capability["evidence"]["experimental_override"] is True
