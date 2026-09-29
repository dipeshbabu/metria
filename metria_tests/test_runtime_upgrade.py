from __future__ import annotations

import copy
import hashlib
import io
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from metria_tests.test_vllm_verify import prefix_case as prefix_case

from metria import PolicyCriterion, VerificationPolicy
from metria.cli import main
from metria.measurements import TokenTrajectoryProtocol
from metria.preparation_upgrade import prepare_vllm_upgrade_recipe
from metria.records import run_record_digest, run_record_from_data
from metria.runtime_environment import ENVIRONMENT_SCHEMA, read_packet, source_digest
from metria.runtimes import vllm
from metria.verification import verify_recipe
from metria.verification_schema import VLLM_UPGRADE_SCOPE
from metria.verification_upgrade import ENVIRONMENTS_KEY, UPGRADE_VERSIONS


@pytest.fixture
def upgrade_case(prefix_case, tmp_path, monkeypatch):
    import metria.runtime_executor as executor
    import metria.runtime_worker as worker

    code = source_digest()
    targets = {}
    for index, (role, version) in enumerate(
        zip(("reference", "candidate"), UPGRADE_VERSIONS, strict=True)
    ):
        prefix = (tmp_path / role).resolve()
        python = prefix / "bin/python"
        python.parent.mkdir(parents=True)
        python.write_bytes(b"same Python interpreter fixture")
        targets[role] = {
            "schema": ENVIRONMENT_SCHEMA,
            "python": str(python),
            "python_sha256": hashlib.sha256(python.read_bytes()).hexdigest(),
            "prefix": str(prefix),
            "metria_sha256": code,
            "runtime": {
                "status": "verified",
                "sha256": str(index + 1) * 64,
                "version": version,
                "dependencies": {
                    "torch": "2.13+cpu",
                    "transformers": "5",
                    "tokenizers": "0.23",
                },
                "file_count": 1,
                "source": "fixture",
            },
            "openmp": None,
            "cpu_affinity": [0, 1, 2, 3],
        }
    original = prefix_case["recipe"]
    model = original.study.runs[0].model
    config = original.measurement_configs[TokenTrajectoryProtocol.name]
    prompts = [
        {**row, "checks": [{"id": "present", "kind": "nonempty"}]}
        for row in config["prompts"]
    ]
    recipe = prepare_vllm_upgrade_recipe(
        Path(model["path"]),
        {"files": model["files"]},
        prompts,
        targets,
        context=256,
        max_tokens=3,
        warmup_trials=0,
        measured_trials=1,
        timeout_s=30,
    )
    case = {
        "recipe": recipe,
        "targets": targets,
        "mode": "normal",
        "calls": [],
        "output": tmp_path / "verification",
    }
    module = vllm._load_vllm()
    active = {"role": "reference"}

    def collective_rpc(engine, method, timeout):
        assert method == "metria_worker_placement"
        assert timeout == 30
        if case["mode"] == "placement_failure":
            raise RuntimeError("native placement observation failed")
        affinity = (
            [1, 2]
            if case["mode"] == "wrong_placement" and active["role"] == "candidate"
            else [0, 1]
        )
        return [
            {
                "device_type": "cpu",
                "cpu_affinity": affinity,
                "affinity_source": "linux_worker_thread_union",
                "torch_intraop_threads": 2,
            }
        ]

    monkeypatch.setattr(module.LLM, "collective_rpc", collective_rpc, raising=False)
    initialize = module.LLM.__init__

    def initialize_with_extension(engine, **kwargs):
        assert (
            kwargs.get("worker_extension_cls")
            == "metria.runtimes.vllm_placement.MetriaWorkerExtension"
        )
        initialize(engine, **kwargs)

    monkeypatch.setattr(module.LLM, "__init__", initialize_with_extension)

    def process(argv, **kwargs):
        path = Path(argv[-1])
        request = read_packet(path)
        role = request["spec"]["environment_selector"]["runtime_environment"]
        case["calls"].append(role)
        if role == "candidate":
            assert (case["output"] / "reference.run.json").exists()
        result = SimpleNamespace(
            returncode=0,
            timed_out=False,
            stdout="private engine output",
            stderr="",
            stdout_truncated=False,
            stderr_truncated=False,
        )
        if role == "candidate" and case["mode"] == "timeout":
            result.returncode, result.timed_out = -9, True
            return result
        active["role"] = role
        target = request["target"]
        module.__version__ = target["runtime"]["version"].split("+", 1)[0]
        with monkeypatch.context() as scoped:
            scoped.setattr(vllm, "_vllm_version", lambda: target["runtime"]["version"])
            scoped.setattr(
                vllm, "require_runtime_pin", lambda *a, **kw: target["runtime"]
            )
            environments = iter((target, {**target, "cpu_affinity": [0, 1]}))
            scoped.setattr(worker, "current_environment", lambda: next(environments))
            scoped.setattr(worker, "bind_cpu", lambda cores: None)
            scoped.setattr(sys, "argv", ["worker", str(path)])
            worker.main()
        receipt = read_packet(path.with_name("receipt.json"))
        if role == "candidate" and case["mode"] == "wrong_nonce":
            receipt["nonce"] = "0" * 48
        if role == "candidate" and case["mode"] == "wrong_record":
            data = read_packet(path.with_name("record.json"))
            data["record"]["run_id"] = "another-run"
            record = run_record_from_data(data)
            path.with_name("record.json").write_text(json.dumps(data))
            receipt["record_digest"] = run_record_digest(record)
        if role == "candidate" and case["mode"] == "wrong_environment":
            receipt["environment"]["prefix"] = "different-prefix"
        path.with_name("receipt.json").write_text(json.dumps(receipt))
        if role == "candidate" and case["mode"] == "cleanup_timeout":
            result.returncode, result.timed_out = -9, True
        return result

    monkeypatch.setattr(executor, "run_process", process)
    return case


def test_upgrade_reuses_normal_execution_with_bound_native_records(upgrade_case):
    result = verify_recipe(upgrade_case["recipe"], upgrade_case["output"])
    data = result.to_data()
    assert result.exit_code == 0, [
        (row["dimension"], row["reason"]) for row in data["comparison"]["issues"]
    ]
    assert data["scope"] == VLLM_UPGRADE_SCOPE and data["verdict"] == "VERIFIED"
    assert data["comparison"]["compatible"] and data["performance"]["available"]
    assert upgrade_case["calls"] == ["reference", "candidate"]
    report = (upgrade_case["output"] / "report.md").read_text()
    assert "0.29.0+cpu -> 0.30.0+cpu" in report
    assert "private" not in report


def test_upgrade_applies_declared_task_policy(upgrade_case):
    recipe = replace(
        upgrade_case["recipe"],
        policy=VerificationPolicy(
            (PolicyCriterion("quality.candidate_pass_rate", "1", minimum=1.0),)
        ),
    )
    result = verify_recipe(recipe, upgrade_case["output"])
    assert result.to_data()["verdict"] == "PASS", result.to_data()


@pytest.mark.parametrize(
    "mode,status",
    [
        ("timeout", "EXECUTION_FAILED"),
        ("wrong_nonce", "EXECUTION_FAILED"),
        ("wrong_record", "EXECUTION_FAILED"),
        ("wrong_environment", "EXECUTION_FAILED"),
        ("wrong_placement", "INSUFFICIENT_EVIDENCE"),
    ],
)
def test_bad_candidate_worker_cannot_hide_behind_valid_reference(
    upgrade_case, mode, status
):
    upgrade_case["mode"] = mode
    result = verify_recipe(upgrade_case["recipe"], upgrade_case["output"])
    assert result.to_data()["verdict"] == status, result.to_data()
    assert (upgrade_case["output"] / "reference.run.json").is_file()
    assert not result.to_data()["performance"]["available"]


def test_changed_interpreter_is_rejected_before_that_worker_starts(upgrade_case):
    Path(upgrade_case["targets"]["candidate"]["python"]).write_bytes(
        b"changed executable"
    )
    result = verify_recipe(upgrade_case["recipe"], upgrade_case["output"])
    assert result.to_data()["verdict"] == "EXECUTION_FAILED"
    assert upgrade_case["calls"] == ["reference"]


def test_cleanup_timeout_retains_completed_native_measurements_without_acceptance(
    upgrade_case,
):
    from metria.records import load_run_record

    upgrade_case["mode"] = "cleanup_timeout"
    result = verify_recipe(upgrade_case["recipe"], upgrade_case["output"])
    assert result.to_data()["verdict"] == "EXECUTION_FAILED"
    record = load_run_record(upgrade_case["output"] / "candidate.run.json")
    assert "trajectory_nonempty_capture_rate" in record.metrics
    assert record.status.value == "timed_out"


def test_native_placement_failure_closes_allocated_engines(upgrade_case, prefix_case):
    upgrade_case["mode"] = "placement_failure"
    result = verify_recipe(upgrade_case["recipe"], upgrade_case["output"])
    assert result.to_data()["verdict"] == "EXECUTION_FAILED"
    assert all(engine.closed for engine in prefix_case["engines"])


def test_shared_deadline_stops_later_runtime_without_starting_it(
    upgrade_case, monkeypatch
):
    import metria.runtime_executor as executor

    times = iter((0.0, 0.0, 31.0))
    monkeypatch.setattr(
        executor, "time", SimpleNamespace(monotonic=lambda: next(times))
    )
    result = verify_recipe(upgrade_case["recipe"], upgrade_case["output"])
    assert result.to_data()["verdict"] == "EXECUTION_FAILED"
    assert upgrade_case["calls"] == ["reference"]


@pytest.mark.parametrize(
    "case",
    [
        "same_prefix",
        "different_python",
        "different_metria",
        "unknown_version",
        "bad_binding",
        "extra_environment",
        "one_run",
        "wrong_role",
        "enabled_cache",
        "changed_context",
        "broad_variation",
    ],
)
def test_upgrade_inputs_reject_unqualified_or_uncontrolled_changes(upgrade_case, case):
    recipe = upgrade_case["recipe"]
    targets = copy.deepcopy(upgrade_case["targets"])
    if case == "same_prefix":
        targets["candidate"]["prefix"] = targets["reference"]["prefix"]
    elif case == "different_python":
        targets["candidate"]["python_sha256"] = "a" * 64
    elif case == "different_metria":
        targets["candidate"]["metria_sha256"] = "a" * 64
    elif case == "unknown_version":
        targets["candidate"]["runtime"]["version"] = "99.0.0"
    elif case == "bad_binding":
        recipe = replace(
            recipe, environment={**recipe.environment, "cpu_binding": [1, 99]}
        )
    elif case == "extra_environment":
        recipe = replace(recipe, environment={**recipe.environment, "extra": True})
    elif case == "one_run":
        recipe = replace(
            recipe, study=replace(recipe.study, runs=recipe.study.runs[:1])
        )
    elif case == "broad_variation":
        recipe = replace(
            recipe,
            study=replace(
                recipe.study,
                comparison=replace(
                    recipe.study.comparison, vary=frozenset({"runtime"})
                ),
            ),
        )
    else:
        left, right = recipe.study.runs
        if case == "wrong_role":
            right = replace(
                right, environment_selector={"runtime_environment": "reference"}
            )
        elif case == "enabled_cache":
            right = replace(
                right, runtime={**right.runtime, "enable_prefix_caching": True}
            )
        elif case == "changed_context":
            right = replace(right, scenario={**right.scenario, "context": 128})
        recipe = replace(recipe, study=replace(recipe.study, runs=(left, right)))
    if case in {
        "same_prefix",
        "different_python",
        "different_metria",
        "unknown_version",
    }:
        recipe = replace(
            recipe, environment={**recipe.environment, ENVIRONMENTS_KEY: targets}
        )
    with pytest.raises(ValueError):
        verify_recipe(recipe, upgrade_case["output"])
    assert not upgrade_case["calls"]


def test_installed_preparation_pins_both_environments_without_launching_models(
    upgrade_case, tmp_path, monkeypatch
):
    import metria.preparation_upgrade as preparation

    monkeypatch.setattr(preparation, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(
        preparation,
        "inspect_environment",
        lambda python: next(
            value
            for value in upgrade_case["targets"].values()
            if value["python"] == str(python)
        ),
    )
    model = upgrade_case["recipe"].study.runs[0].model
    descriptor = tmp_path / "descriptor.json"
    descriptor.write_text(json.dumps({"files": dict(model["files"])}))
    workload = tmp_path / "workload.jsonl"
    workload.write_text(json.dumps({"id": "task", "prompt": "test"}) + "\n")
    output = tmp_path / "study.json"
    arguments = [
        "recipe",
        "prepare-vllm-upgrade",
        "--reference-python",
        upgrade_case["targets"]["reference"]["python"],
        "--candidate-python",
        upgrade_case["targets"]["candidate"]["python"],
        "--model",
        model["path"],
        "--descriptor",
        str(descriptor),
        "--workload",
        str(workload),
        "--output",
        str(output),
    ]
    out, err = io.StringIO(), io.StringIO()
    assert main(arguments, stdout=out, stderr=err) == 0, err.getvalue()
    assert output.is_file() and not upgrade_case["calls"]
    assert main(arguments, stdout=out, stderr=err) == 2
