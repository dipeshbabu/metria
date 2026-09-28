from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from metria import (
    ComparisonPlan,
    RunSpec,
    StudyRecipe,
    StudySpec,
    verification,
    verification_worker,
)
from metria.measurements import TokenTrajectoryProtocol, TrajectoryAgreementAnalysis
from metria.processes import ProcessResult
from metria.records import load_run_record
from metria.runtimes import vllm
from metria.verification_vllm import VLLM_VARIATIONS, build_route, trial_identity


@pytest.fixture
def prefix_case(tmp_path, monkeypatch):
    model = tmp_path / "pinned model [data]"
    model.mkdir()
    pins = {}
    for name in (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "model.safetensors",
    ):
        content = ("fixture " + name).encode()
        (model / name).write_bytes(content)
        pins[name] = hashlib.sha256(content).hexdigest()
    config = {
        "prompts": [
            {"id": "a", "prompt": "private shared prefix a"},
            {"id": "b", "prompt": "private shared prefix b"},
        ],
        "warmup_trials": 1,
        "measured_trials": 2,
    }
    runs = tuple(
        RunSpec(
            model={"path": str(model), "files": pins},
            runtime={
                "name": "vllm",
                "version": "0.30.0+cpu",
                "dtype": "float32",
                "gpu_memory_utilization": 0.5,
                "max_num_seqs": 1,
                "max_model_len": 256,
                "tensor_parallel_size": 1,
                "enforce_eager": True,
                "enable_prefix_caching": enabled,
                "trust_remote_code": False,
            },
            scenario={
                "context": 128,
                "max_tokens": 3,
                "seed": 42,
                "temperature": 0.0,
                "chat_template": False,
            },
            measurements=(TokenTrajectoryProtocol.name,),
            trial_policy=trial_identity(config),
        )
        for enabled in (False, True)
    )
    recipe = StudyRecipe(
        study=StudySpec(
            name="prefix fixture",
            runs=runs,
            comparison=ComparisonPlan(
                vary=VLLM_VARIATIONS,
                control=frozenset(
                    {"model", "scenario", "measurements", "trial_policy"}
                ),
                analyses=(TrajectoryAgreementAnalysis.name,),
            ),
        ),
        measurement_configs={TokenTrajectoryProtocol.name: config},
        environment={
            "vllm_distribution_sha256": "a" * 64,
            "verification_timeout_s": 30,
        },
    )
    case = {"recipe": recipe, "mode": "normal", "engines": []}

    class Engine:
        def __init__(self, **kwargs):
            self.enabled = kwargs["enable_prefix_caching"]
            self.seen = False
            self.closed = False
            self.model_config = SimpleNamespace(
                model=kwargs["model"],
                tokenizer=kwargs["model"],
                max_model_len=kwargs["max_model_len"],
                dtype=kwargs["dtype"],
                revision=kwargs.get("revision"),
                tokenizer_revision=kwargs.get("tokenizer_revision"),
            )
            cache = SimpleNamespace(
                cache_dtype=kwargs["kv_cache_dtype"],
                gpu_memory_utilization=kwargs["gpu_memory_utilization"],
                enable_prefix_caching=False
                if case["mode"] == "ignored"
                else self.enabled,
            )
            self.llm_engine = SimpleNamespace(
                vllm_config=SimpleNamespace(
                    cache_config=cache,
                    parallel_config=SimpleNamespace(tensor_parallel_size=1),
                )
            )
            case["engines"].append(self)

        def get_tokenizer(self):
            return SimpleNamespace(
                name_or_path=self.model_config.model, chat_template=None, init_kwargs={}
            )

        def reset_prefix_cache(self):
            self.seen = False
            return case["mode"] != "reset_failed"

        def generate(self, prompts, **kwargs):
            if self.enabled and case["mode"] == "inference_failed":
                raise RuntimeError("private runtime failure")
            cached = (
                16 if self.enabled and self.seen and case["mode"] != "no_hits" else 0
            )
            self.seen = True
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(text="private completion", token_ids=[1, 2, 3])
                    ],
                    prompt_token_ids=list(range(32)),
                    num_cached_tokens=None
                    if case["mode"] == "missing_counts"
                    else cached,
                    finished=True,
                )
                for _ in prompts
            ]

        def shutdown(self):
            self.closed = True

    module = SimpleNamespace(
        __version__="0.30.0",
        LLM=Engine,
        SamplingParams=lambda **kw: SimpleNamespace(**kw),
    )
    monkeypatch.setattr(
        vllm,
        "native_hardware",
        lambda: {"status": "observed", "device_type": "cpu", "source": "fixture"},
    )
    monkeypatch.setattr(vllm, "_vllm_available", lambda: True)
    monkeypatch.setattr(vllm, "_vllm_version", lambda: "0.30.0+cpu")
    monkeypatch.setattr(vllm, "_load_vllm", lambda: module)
    monkeypatch.setattr(
        vllm,
        "require_runtime_pin",
        lambda *a, **kw: {
            "status": "verified",
            "sha256": "a" * 64,
            "version": "0.30.0+cpu",
        },
    )
    return case


def execute(case, output, **kwargs):
    return verification._verify_with_profile(
        case["recipe"], output, build_route(case["recipe"]), **kwargs
    )


def test_complete_prefix_verification_retains_native_evidence(prefix_case, tmp_path):
    result = execute(prefix_case, tmp_path / "verified")
    assert result.exit_code == 0, result.to_data()
    assert result.manifest["comparison_status"] == "VALID"
    assert result.manifest["performance"]["available"] is True
    assert (
        result.manifest["performance"]["metric"]["method"]
        == "metria.runtime_call_latency"
    )
    assert all(engine.closed for engine in prefix_case["engines"])
    report = (result.output_dir / "report.md").read_text()
    assert "Prefix caching: False -> True" in report
    assert "Loaded-engine request latency" in report
    assert "private" not in report
    candidate = load_run_record(result.output_dir / "candidate.run.json")
    assert candidate.observed["artifacts"]["model"]["status"] == "verified"
    assert any(row["cached_tokens"] for row in candidate.observed["invocations"])


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("ignored", 5),
        ("inference_failed", 5),
        ("reset_failed", 5),
        ("missing_counts", 4),
        ("no_hits", 4),
    ],
)
def test_bad_native_evidence_never_produces_success(
    prefix_case, tmp_path, mode, expected
):
    prefix_case["mode"] = mode
    result = execute(prefix_case, tmp_path / mode)
    assert result.exit_code == expected
    assert (result.output_dir / "reference.run.json").is_file()
    assert (result.output_dir / "candidate.run.json").is_file()
    assert result.manifest["acceptance_policy_evaluated"] is False


def test_undeclared_runtime_change_fails_before_launch(prefix_case, tmp_path):
    recipe = prefix_case["recipe"]
    left, right = recipe.study.runs
    right = replace(right, runtime={**right.runtime, "max_model_len": 512})
    recipe = replace(recipe, study=replace(recipe.study, runs=(left, right)))
    with pytest.raises(ValueError, match="only"):
        verification.verify_recipe(recipe, tmp_path / "invalid")
    assert not prefix_case["engines"]


def test_supervisor_accepts_only_bound_worker_records(
    prefix_case, tmp_path, monkeypatch
):
    def invoke(recipe, output):
        result = execute(prefix_case, output, reserved_output=True, defer_result=True)
        return ProcessResult(
            "fixture", result.exit_code, "private output", "private error", False, 0.1
        )

    monkeypatch.setattr(verification_worker, "_invoke_worker", invoke)
    result = verification.verify_recipe(prefix_case["recipe"], tmp_path / "isolated")
    assert result.exit_code == 0
    assert not (result.output_dir / "worker-result.json").exists()
    assert "private" not in (result.output_dir / "verification.json").read_text()


@pytest.mark.parametrize(
    "kind,code", [("timeout", 5), ("interrupt", 130), ("bad_receipt", 5)]
)
def test_supervisor_retains_both_failure_records(
    prefix_case, tmp_path, monkeypatch, kind, code
):
    def invoke(recipe, output):
        if kind == "interrupt":
            raise KeyboardInterrupt("private interrupt")
        if kind == "bad_receipt":
            (output / "worker-result.json").write_text(json.dumps({"schema": "wrong"}))
        return ProcessResult(
            "fixture", 1, "private output", "private stderr", kind == "timeout", 0.1
        )

    monkeypatch.setattr(verification_worker, "_invoke_worker", invoke)
    result = verification.verify_recipe(prefix_case["recipe"], tmp_path / kind)
    assert result.exit_code == code
    assert result.manifest["verdict"] == "EXECUTION_FAILED"
    assert result.manifest["comparison_status"] == "NOT_EVALUATED"
    assert not result.manifest["analyses"]
    assert (result.output_dir / "reference.run.json").is_file()
    assert (result.output_dir / "candidate.run.json").is_file()
    assert "private" not in (result.output_dir / "verification.json").read_text()


def test_real_worker_transport_fails_closed_without_a_matching_runtime(
    prefix_case, tmp_path
):
    recipe = prefix_case["recipe"]
    recipe = replace(
        recipe, environment={**recipe.environment, "verification_timeout_s": 3}
    )
    result = verification.verify_recipe(recipe, tmp_path / "real worker [spaces]")
    assert result.exit_code == 5
    assert result.manifest["execution_boundary"]["mode"] == "bounded_native_worker"
    assert (result.output_dir / "reference.run.json").is_file()
    assert (result.output_dir / "candidate.run.json").is_file()
    assert not (result.output_dir / ".worker-recipe.json").exists()


def test_supervisor_keeps_completed_reference_after_worker_timeout(
    prefix_case, tmp_path, monkeypatch
):
    digest = None

    def invoke(recipe, output):
        nonlocal digest
        result = execute(prefix_case, output, reserved_output=True, defer_result=True)
        digest = result.manifest["records"]["reference"]["record_digest"]
        (output / "candidate.run.json").unlink()
        return ProcessResult("fixture", 1, "", "", True, 0.1)

    monkeypatch.setattr(verification_worker, "_invoke_worker", invoke)
    result = verification.verify_recipe(prefix_case["recipe"], tmp_path / "partial")
    assert result.exit_code == 5
    assert result.manifest["records"]["reference"]["record_digest"] == digest
    assert result.manifest["records"]["candidate"]["status"] == "timed_out"


def test_output_collision_never_launches_a_worker(prefix_case, tmp_path, monkeypatch):
    marker = tmp_path / "keep.txt"
    marker.write_text("keep")
    monkeypatch.setattr(
        verification_worker,
        "_invoke_worker",
        lambda *a: pytest.fail("worker must not launch"),
    )
    with pytest.raises(FileExistsError):
        verification.verify_recipe(prefix_case["recipe"], tmp_path)
    assert marker.read_text() == "keep"


def test_worker_entrypoint_defers_publication_until_parent_validation(
    prefix_case, tmp_path, monkeypatch
):
    from metria.recipes import study_recipe_digest, study_recipe_to_json

    output = tmp_path / "worker entry"
    output.mkdir()
    transport = output / ".worker-recipe.json"
    transport.write_text(study_recipe_to_json(prefix_case["recipe"]), encoding="utf-8")
    argv = [
        "worker",
        "--recipe",
        str(transport),
        "--output",
        str(output),
        "--digest",
        study_recipe_digest(prefix_case["recipe"]),
    ]
    monkeypatch.setattr(verification_worker.sys, "argv", argv)
    assert verification_worker.main() == 0
    assert (output / "worker-result.json").is_file()
    assert not (output / "verification.json").exists()
    assert not (output / "report.md").exists()
    monkeypatch.setattr(verification_worker.sys, "argv", [*argv[:-1], "wrong digest"])
    with pytest.raises(ValueError, match="digest"):
        verification_worker.main()


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", "unqualified"),
        ("trust_remote_code", True),
        ("max_num_seqs", 2),
        ("tensor_parallel_size", 2),
        ("enforce_eager", False),
        ("dtype", "unknown"),
    ],
)
def test_unqualified_runtime_configuration_fails_before_launch(
    prefix_case, tmp_path, field, value
):
    recipe = prefix_case["recipe"]
    runs = tuple(
        replace(run, runtime={**run.runtime, field: value}) for run in recipe.study.runs
    )
    recipe = replace(recipe, study=replace(recipe.study, runs=runs))
    with pytest.raises((ValueError, TypeError)):
        verification.verify_recipe(recipe, tmp_path / "invalid runtime")
    assert not prefix_case["engines"]


@pytest.mark.parametrize(
    "environment",
    [
        {"vllm_distribution_sha256": "bad", "verification_timeout_s": 30},
        {"vllm_distribution_sha256": "a" * 64, "verification_timeout_s": True},
        {"vllm_distribution_sha256": "a" * 64, "verification_timeout_s": 0},
        {"vllm_distribution_sha256": "a" * 64, "verification_timeout_s": float("inf")},
    ],
)
def test_invalid_worker_environment_rejected_before_launch(
    prefix_case, tmp_path, environment
):
    recipe = replace(prefix_case["recipe"], environment=environment)
    with pytest.raises((ValueError, TypeError)):
        verification.verify_recipe(recipe, tmp_path / "invalid environment")
    assert not prefix_case["engines"]


@pytest.mark.parametrize(
    "damage", ["model", "runtime", "applied", "hardware", "invocations", "resets"]
)
def test_required_evidence_cannot_be_removed_from_completed_records(
    prefix_case, tmp_path, damage
):
    from metria.recipes import _json_value

    result = execute(prefix_case, tmp_path / damage)
    record = load_run_record(result.output_dir / "candidate.run.json")
    observed = _json_value(record.observed, path="observed")
    if damage in {"model", "runtime"}:
        observed["artifacts"][damage]["status"] = "unknown"
    elif damage == "applied":
        observed["applied"]["status"] = "unverified"
    elif damage == "hardware":
        observed["hardware"]["status"] = "unknown"
    elif damage == "invocations":
        observed["invocations"] = []
    else:
        observed["reset_events"] = None
    damaged = replace(record, observed=observed)
    assert build_route(prefix_case["recipe"]).evidence_gaps(damaged)


def test_performance_rejects_an_uncontrolled_hardware_change(prefix_case, tmp_path):
    from metria.measurements.prefix_performance import compare_prefix_performance

    result = execute(prefix_case, tmp_path / "hardware mismatch")
    left = load_run_record(result.output_dir / "reference.run.json")
    right = load_run_record(result.output_dir / "candidate.run.json")
    right = replace(
        right,
        observed={
            **right.observed,
            "hardware": {
                "status": "observed",
                "device_type": "cuda",
                "name": "different fixture",
            },
        },
    )
    assert compare_prefix_performance(left, right, True)["available"] is False


def test_preparation_pins_runtime_and_model_without_loading_engine(
    prefix_case, monkeypatch
):
    from metria import preparation

    monkeypatch.setattr(
        preparation,
        "installed_runtime_identity",
        lambda: {"version": "0.30.0+cpu", "sha256": "a" * 64},
    )
    old = prefix_case["recipe"]
    model = old.study.runs[0].model
    config = old.measurement_configs[TokenTrajectoryProtocol.name]
    prepared = preparation.prepare_vllm_prefix_recipe(
        model["path"],
        {"files": model["files"]},
        config["prompts"],
        warmup_trials=0,
        measured_trials=4,
    )
    assert prepared.study.runs[0].runtime["enable_prefix_caching"] is False
    assert prepared.study.runs[1].runtime["enable_prefix_caching"] is True
    assert prepared.study.runs[0].trial_policy["measured_trials"] == 4
    assert prepared.environment["vllm_distribution_sha256"] == "a" * 64
    assert not prefix_case["engines"]
