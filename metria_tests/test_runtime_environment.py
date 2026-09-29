from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from metria import runtime_environment as environments


def _target(tmp_path):
    prefix = (tmp_path / "environment").resolve()
    python = prefix / "bin/python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"interpreter fixture")
    return {
        "schema": environments.ENVIRONMENT_SCHEMA,
        "python": str(python),
        "python_sha256": hashlib.sha256(python.read_bytes()).hexdigest(),
        "prefix": str(prefix),
        "metria_sha256": "a" * 64,
        "runtime": {"status": "verified", "version": "0.29.0+cpu", "sha256": "b" * 64},
        "openmp": None,
        "cpu_affinity": [0, 1],
    }


def test_environment_inspection_uses_bound_receipt_not_console_output(
    tmp_path, monkeypatch
):
    target = _target(tmp_path)
    monkeypatch.setattr(environments, "source_digest", lambda: "a" * 64)

    def run(argv, **kwargs):
        request = Path(argv[-1])
        payload, digest = environments.read_bound_packet(request)
        environments.write_private(
            request.with_name("receipt.json"),
            {
                "schema": environments.RECEIPT_SCHEMA,
                "nonce": payload["nonce"],
                "request_sha256": digest,
                "environment": target,
            },
        )
        return SimpleNamespace(
            returncode=0, timed_out=False, stdout="library log, not JSON", stderr=""
        )

    monkeypatch.setattr(environments, "run_process", run)
    assert environments.inspect_environment(target["python"]) == target


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "unknown"),
        ("python", "relative"),
        ("metria_sha256", "b" * 64),
        ("python_sha256", "bad"),
        ("runtime", {}),
        ("openmp", {"path": "relative", "sha256": "a" * 64}),
        ("cpu_affinity", [True, 1]),
        ("cpu_affinity", [1, 0]),
        ("cpu_affinity", []),
    ],
)
def test_incomplete_or_ambiguous_environment_is_rejected(tmp_path, field, value):
    target = _target(tmp_path)
    target[field] = value
    with pytest.raises(ValueError):
        environments.validate_environment(target, metria_sha256="a" * 64)


def test_child_environment_binds_preloaded_library_and_removes_import_overrides(
    tmp_path, monkeypatch
):
    target = _target(tmp_path)
    library = tmp_path / "libiomp5.so"
    library.write_bytes(b"pinned library")
    target["openmp"] = {
        "path": str(library.resolve()),
        "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    }
    monkeypatch.setenv("PYTHONPATH", "untrusted-path")
    monkeypatch.setenv("PYTHONHOME", "untrusted-prefix")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    result = environments.child_environment(
        Path(target["python"]), descriptor=target, binding="0-1"
    )
    assert "PYTHONPATH" not in result and "PYTHONHOME" not in result
    assert "VLLM_ALLOW_INSECURE_SERIALIZATION" not in result
    assert result["LD_PRELOAD"] == str(library.resolve())
    assert result["VLLM_CPU_OMP_THREADS_BIND"] == "0-1"
    assert result["HF_HUB_OFFLINE"] == "1"
    library.write_bytes(b"changed library")
    with pytest.raises(ValueError, match="changed"):
        environments.child_environment(Path(target["python"]), descriptor=target)


@pytest.mark.parametrize(
    "text", ['{"duplicate":1,"duplicate":2}', '{"value":NaN}', "[]"]
)
def test_worker_messages_require_strict_json_objects(tmp_path, text):
    path = tmp_path / "message.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        environments.read_packet(path)


def test_private_packet_is_bounded_and_never_overwrites(tmp_path):
    path = tmp_path / "request.json"
    expected = environments.write_private(path, {"value": "data"})
    payload, digest = environments.read_bound_packet(path)
    assert payload == {"value": "data"} and digest == expected
    with pytest.raises(FileExistsError):
        environments.write_private(path, {})
    with pytest.raises(ValueError, match="size limit"):
        environments.read_packet(path, 2)


def test_failed_environment_probe_has_actionable_setup_error(tmp_path, monkeypatch):
    target = _target(tmp_path)
    monkeypatch.setattr(
        environments,
        "run_process",
        lambda *a, **kw: SimpleNamespace(returncode=1, timed_out=False),
    )
    with pytest.raises(ValueError, match="install the same Metria"):
        environments.inspect_environment(target["python"])


def test_current_environment_records_real_interpreter_and_openmp_bytes(
    tmp_path, monkeypatch
):
    target = _target(tmp_path)
    library = Path(target["prefix"]) / "lib/libiomp5.so"
    library.parent.mkdir()
    library.write_bytes(b"openmp fixture")
    monkeypatch.setattr(environments.sys, "executable", target["python"])
    monkeypatch.setattr(environments.sys, "prefix", target["prefix"])
    monkeypatch.setattr(environments, "source_digest", lambda: "a" * 64)
    monkeypatch.setattr(
        environments, "installed_runtime_identity", lambda: target["runtime"]
    )
    monkeypatch.setattr(
        environments.os, "sched_getaffinity", lambda pid: {0, 1}, raising=False
    )
    observed = environments.current_environment()
    assert observed["python_sha256"] == target["python_sha256"]
    assert (
        observed["openmp"]["sha256"] == hashlib.sha256(library.read_bytes()).hexdigest()
    )
    environments.validate_environment(observed, metria_sha256="a" * 64)
    assert environments.child_environment(Path(target["python"]))["LD_PRELOAD"] == str(
        library
    )


def test_inspection_worker_returns_its_observation_without_running_inference(
    tmp_path, monkeypatch
):
    import metria.runtime_worker as worker

    target = _target(tmp_path)
    monkeypatch.setattr(worker, "current_environment", lambda: target)
    assert worker.execute_request(
        {"metria_sha256": "a" * 64, "mode": "inspect"}, tmp_path
    ) == {"environment": target}


@pytest.mark.parametrize(
    "payload",
    [
        {
            "schema": "unknown",
            "mode": "inspect",
            "nonce": "test",
            "metria_sha256": "a" * 64,
        },
        {
            "schema": environments.REQUEST_SCHEMA,
            "mode": "inspect",
            "nonce": "test",
            "metria_sha256": "b" * 64,
        },
    ],
)
def test_worker_rejects_protocol_or_implementation_mismatch(
    tmp_path, monkeypatch, payload
):
    import sys

    import metria.runtime_worker as worker

    path = tmp_path / "request.json"
    environments.write_private(path, payload)
    monkeypatch.setattr(sys, "argv", ["worker", str(path)])
    monkeypatch.setattr(worker, "source_digest", lambda: "a" * 64)
    with pytest.raises(ValueError):
        worker.main()
    assert not path.with_name("receipt.json").exists()


def test_native_worker_placement_uses_worker_observations(monkeypatch):
    import os
    import sys

    from metria.runtimes import vllm_placement

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(get_num_threads=lambda: 2)
    )
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {1, 0}, raising=False)
    monkeypatch.setattr(vllm_placement, "worker_affinity", lambda: [0, 1])
    assert vllm_placement.observe_worker(
        SimpleNamespace(device=SimpleNamespace(type="cpu"))
    ) == {
        "device_type": "cpu",
        "cpu_affinity": [0, 1],
        "affinity_source": "linux_worker_thread_union",
        "torch_intraop_threads": 2,
    }


def test_named_worker_extension_needs_no_callable_serialization(monkeypatch):
    from metria.runtimes import vllm_placement

    monkeypatch.setattr(
        vllm_placement, "observe_worker", lambda worker: {"device_type": "cpu"}
    )
    assert vllm_placement.MetriaWorkerExtension().metria_worker_placement() == {
        "device_type": "cpu"
    }


def test_worker_affinity_observes_all_threads_and_tolerates_exited_threads(
    tmp_path, monkeypatch
):
    import os

    from metria.runtimes.vllm_placement import worker_affinity

    for name in ("11", "12", "13"):
        (tmp_path / name).mkdir()

    def affinity(pid):
        if pid == 13:
            raise ProcessLookupError()
        return {0} if pid == 11 else {1}

    monkeypatch.setattr(os, "sched_getaffinity", affinity, raising=False)
    assert worker_affinity(tmp_path) == [0, 1]
    assert worker_affinity(tmp_path / "missing") is None


def test_controlled_binding_is_applied_before_worker_inference(tmp_path, monkeypatch):
    import metria.runtime_worker as worker

    tasks = tmp_path / "tasks"
    tasks.mkdir()
    for name in ("11", "12"):
        (tasks / name).mkdir()
    original = worker.Path
    monkeypatch.setattr(
        worker,
        "Path",
        lambda value: tasks if value == "/proc/self/task" else original(value),
    )
    bound = []
    monkeypatch.setattr(
        worker.os,
        "sched_setaffinity",
        lambda pid, cpus: bound.append((pid, cpus)),
        raising=False,
    )
    worker.bind_cpu((0, 1))
    assert {pid for pid, cpus in bound} == {0, 11, 12}
    assert all(cpus == {0, 1} for pid, cpus in bound)


def test_controlled_worker_rejects_host_without_cpu_affinity(monkeypatch):
    import metria.runtime_worker as worker

    monkeypatch.delattr(worker.os, "sched_setaffinity", raising=False)
    with pytest.raises(ValueError, match="Linux CPU affinity"):
        worker.bind_cpu((0, 1))


@pytest.mark.parametrize("wrong", ["type", "study", "run", "spec"])
def test_study_executor_never_persists_evidence_for_another_run(wrong):
    from metria import ComparisonPlan, RunRecord, RunSpec, RunStatus, StudySpec
    from metria.study_execution import execute_study

    spec = RunSpec(
        model={"id": "fixture"},
        runtime={"name": "fixture"},
        scenario={},
        measurements=("fixture",),
    )
    study = StudySpec(name="bound-study", runs=(spec,), comparison=ComparisonPlan())
    saved = []

    def execute(**kwargs):
        if wrong == "type":
            return object()
        return RunRecord(
            study_name="other" if wrong == "study" else kwargs["study_name"],
            run_id="other" if wrong == "run" else kwargs["run_id"],
            requested=RunSpec(
                model={"id": "other"}, runtime={"name": "fixture"}, scenario={}
            )
            if wrong == "spec"
            else spec,
            resolved={},
            observed={},
            status=RunStatus.COMPLETED,
        )

    with pytest.raises((TypeError, ValueError)):
        execute_study(
            study,
            adapters={"fixture": SimpleNamespace(name="fixture")},
            measurements={"fixture": SimpleNamespace(name="fixture")},
            measurement_configs={},
            environment={},
            record_sink=saved.append,
            _run_executor=execute,
        )
    assert not saved
