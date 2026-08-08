from __future__ import annotations

import io
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from metria import (
    ComparisonPlan,
    HardwareFingerprint,
    RunSpec,
    StudyRecipe,
    StudySpec,
    dump_study_recipe,
    load_run_record,
    study_recipe_digest,
)
from metria.cli import PLUGIN_REPORT_SCHEMA, main
from metria.protocols import (
    CaptureRequest,
    InferenceBatch,
    InferenceRequest,
    MeasurementResult,
    SupportReport,
)
from metria.registry import (
    PluginAvailability,
    PluginDescriptor,
    PluginKind,
    RegistryBundle,
)
from metria.runner import STUDY_RESULT_SCHEMA, execute_recipe_to_directory


class _Session:
    def __init__(self) -> None:
        self.closed = False

    def infer(
        self,
        requests: Sequence[InferenceRequest],
        capture: Sequence[CaptureRequest] = (),
    ) -> InferenceBatch:
        del capture
        if self.closed:
            raise RuntimeError("session is closed")
        return InferenceBatch(outputs=tuple("ok" for _ in requests))

    def reset(self, scope: str = "measurement") -> None:
        del scope
        if self.closed:
            raise RuntimeError("session is closed")

    def close(self) -> None:
        self.closed = True


class _Adapter:
    name = "test-runtime"

    def probe(
        self,
        spec: RunSpec,
        environment: Mapping[str, Any],
    ) -> SupportReport:
        del spec, environment
        return SupportReport(status="supported", evidence={"runtime": self.name})

    def resolve(
        self,
        spec: RunSpec,
        environment: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del environment
        return {
            "runtime": {"name": self.name, "version": "1"},
            "model": dict(spec.model),
        }

    def launch(
        self,
        resolved: Mapping[str, Any],
        environment: Mapping[str, Any],
    ) -> _Session:
        del resolved, environment
        return _Session()

    def observe(self, session: _Session) -> Mapping[str, Any]:
        return {"runtime": self.name, "closed": session.closed}


class _Measurement:
    name = "test.measurement"
    version = "1"

    def requirements(self, config: Mapping[str, Any]) -> tuple[CaptureRequest, ...]:
        del config
        return ()

    def execute(
        self,
        session: _Session,
        scenario: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> MeasurementResult:
        del session, scenario, config
        return MeasurementResult(evidence={"method": self.name})


class _FailingMeasurement(_Measurement):
    def execute(
        self,
        session: _Session,
        scenario: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> MeasurementResult:
        del session, scenario, config
        raise RuntimeError("synthetic measurement failure")


def _registry(
    *,
    available: bool = True,
    failing: bool = False,
) -> RegistryBundle:
    measurement = _FailingMeasurement() if failing else _Measurement()
    return RegistryBundle(
        runtimes={"test-runtime": _Adapter()},
        measurements={measurement.name: measurement},
        analyses={},
        descriptors=(
            PluginDescriptor(
                name="test-runtime",
                kind=PluginKind.RUNTIME,
                availability=(
                    PluginAvailability.AVAILABLE
                    if available
                    else PluginAvailability.UNAVAILABLE
                ),
                reason=None if available else "runtime intentionally unavailable",
            ),
            PluginDescriptor(
                name=measurement.name,
                kind=PluginKind.MEASUREMENT,
                availability=PluginAvailability.AVAILABLE,
                version=measurement.version,
            ),
        ),
    )


def _recipe() -> StudyRecipe:
    return StudyRecipe(
        study=StudySpec(
            name="runner-study",
            runs=(
                RunSpec(
                    model={"id": "example/model", "revision": "abc123"},
                    runtime={"name": "test-runtime"},
                    scenario={"name": "decode"},
                    measurements=("test.measurement",),
                ),
            ),
            comparison=ComparisonPlan(),
        ),
        measurement_configs={"test.measurement": {}},
        environment={"requested_pool": "test"},
    )


def _hardware() -> HardwareFingerprint:
    return HardwareFingerprint(
        platform={"system": "TestOS", "machine": "unit"},
        host={"hostname_sha256": "a" * 64, "cpu_count": 4},
        software={"python_version": "3.test"},
    )


def test_execute_recipe_persists_records_manifest_and_invocation_provenance(
    tmp_path: Path,
) -> None:
    recipe = _recipe()
    output = tmp_path / "result"

    persisted = execute_recipe_to_directory(
        recipe,
        output_dir=output,
        registry=_registry(),
        hardware=_hardware(),
    )

    assert persisted.successful is True
    assert persisted.manifest_path == output / "study-result.json"
    assert persisted.run_paths == (output / "run-0000.json",)
    record = load_run_record(persisted.run_paths[0])
    invocation = record.provenance["invocation"]
    assert invocation["recipe"]["digest"] == study_recipe_digest(recipe)
    assert invocation["hardware"]["platform"]["system"] == "TestOS"
    manifest = json.loads(persisted.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == STUDY_RESULT_SCHEMA
    assert manifest["successful"] is True
    assert manifest["records"][0]["path"] == "run-0000.json"
    assert len(manifest["records"][0]["record_digest"]) == 64
    assert len(manifest["records"][0]["evidence_digest"]) == 64


def test_fixed_inputs_produce_deterministic_study_manifest(tmp_path: Path) -> None:
    recipe = _recipe()
    first = execute_recipe_to_directory(
        recipe,
        output_dir=tmp_path / "first",
        registry=_registry(),
        hardware=_hardware(),
    )
    second = execute_recipe_to_directory(
        recipe,
        output_dir=tmp_path / "second",
        registry=_registry(),
        hardware=_hardware(),
    )

    assert first.manifest_path.read_text(
        encoding="utf-8"
    ) == second.manifest_path.read_text(encoding="utf-8")


def test_unavailable_runtime_fails_before_output_directory_is_created(
    tmp_path: Path,
) -> None:
    recipe_path = tmp_path / "study.json"
    dump_study_recipe(recipe_path, _recipe())
    output = tmp_path / "result"
    stderr = io.StringIO()

    status = main(
        ["run", str(recipe_path), "--output-dir", str(output)],
        stdout=io.StringIO(),
        stderr=stderr,
        registry=_registry(available=False),
    )

    assert status == 2
    assert "is unavailable" in stderr.getvalue()
    assert not output.exists()


def test_failed_measurement_is_persisted_and_cli_returns_one(tmp_path: Path) -> None:
    recipe_path = tmp_path / "study.json"
    dump_study_recipe(recipe_path, _recipe())
    output = tmp_path / "result"
    stdout = io.StringIO()

    status = main(
        [
            "run",
            str(recipe_path),
            "--output-dir",
            str(output),
            "--json",
        ],
        stdout=stdout,
        stderr=io.StringIO(),
        registry=_registry(failing=True),
    )
    payload = json.loads(stdout.getvalue())
    record = load_run_record(output / "run-0000.json")
    manifest = json.loads((output / "study-result.json").read_text(encoding="utf-8"))

    assert status == 1
    assert payload["successful"] is False
    assert record.status.value == "failed"
    assert manifest["successful"] is False
    assert manifest["records"][0]["status"] == "failed"


def test_nonempty_output_directory_is_rejected_before_execution(tmp_path: Path) -> None:
    output = tmp_path / "result"
    output.mkdir()
    (output / "existing.txt").write_text("keep", encoding="utf-8")

    try:
        execute_recipe_to_directory(
            _recipe(),
            output_dir=output,
            registry=_registry(),
            hardware=_hardware(),
        )
    except ValueError as exc:
        assert "must be empty" in str(exc)
    else:
        raise AssertionError("non-empty output directory should have been rejected")
    assert (output / "existing.txt").read_text(encoding="utf-8") == "keep"


def test_plugins_cli_lists_explicit_registry_metadata() -> None:
    stdout = io.StringIO()

    status = main(
        ["plugins", "--json"],
        stdout=stdout,
        stderr=io.StringIO(),
        registry=_registry(),
    )
    payload = json.loads(stdout.getvalue())

    assert status == 0
    assert payload["schema"] == PLUGIN_REPORT_SCHEMA
    assert payload["plugins"] == [
        {
            "availability": "available",
            "kind": "measurement",
            "name": "test.measurement",
            "reason": None,
            "version": "1",
        },
        {
            "availability": "available",
            "kind": "runtime",
            "name": "test-runtime",
            "reason": None,
            "version": None,
        },
    ]
