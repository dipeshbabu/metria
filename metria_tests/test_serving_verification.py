import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from metria_tests.test_vllm_verify import prefix_case  # noqa: F401

from metria import PolicyCriterion, VerificationPolicy, verification
from metria.measurements import TokenTrajectoryProtocol, TrajectoryAgreementAnalysis
from metria.measurements.serving_impact import ServingImpactAnalysis, compare_serving
from metria.measurements.verification_impact import VerificationImpactAnalysis
from metria.preparation_serving import prepare_serving_recipe
from metria.records import load_run_record
from metria.runtimes import vllm
from metria.runtimes.vllm_serving import AsyncEngineFacade
from metria.verification_serving import (
    SERVING_CONTROLS,
    SERVING_VARIATIONS,
    build_route,
    evidence_gaps,
    trial_identity,
)


@pytest.fixture
def serving_case(prefix_case, monkeypatch):  # noqa: F811
    case = prefix_case
    original = case["recipe"]
    config = dict(original.measurement_configs[TokenTrajectoryProtocol.name])
    config["prompts"] = [
        {**row, "checks": [{"id": "present", "kind": "nonempty"}]}
        for row in config["prompts"]
    ]
    runs = tuple(
        replace(
            run,
            runtime={**run.runtime, "max_num_seqs": 2, "enable_prefix_caching": False},
            trial_policy=trial_identity(config, concurrency),
        )
        for run, concurrency in zip(original.study.runs, (1, 2), strict=True)
    )
    case["recipe"] = replace(
        original,
        study=replace(
            original.study,
            runs=runs,
            comparison=replace(
                original.study.comparison,
                vary=SERVING_VARIATIONS,
                control=SERVING_CONTROLS,
                analyses=(
                    TrajectoryAgreementAnalysis.name,
                    VerificationImpactAnalysis.name,
                    ServingImpactAnalysis.name,
                ),
            ),
        ),
        measurement_configs={TokenTrajectoryProtocol.name: config},
    )
    engine_type = vllm._load_vllm().LLM

    class Native:
        def __init__(self, kwargs):
            self.base = engine_type(**kwargs)
            self.model_config = self.base.model_config
            self.vllm_config = self.base.llm_engine.vllm_config
            self.vllm_config.scheduler_config = SimpleNamespace(
                max_num_seqs=1 if case["mode"] == "capacity" else kwargs["max_num_seqs"]
            )

        def get_tokenizer(self):
            return self.base.get_tokenizer()

        async def generate(self, prompt, params, request_id):
            for index in range(1, 4):
                await asyncio.sleep(0)
                yield SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            text="private completion", token_ids=list(range(index))
                        )
                    ],
                    prompt_token_ids=list(range(32)),
                    num_cached_tokens=0,
                    finished=index == 3,
                )

        async def collective_rpc(self, method, timeout):
            return [{"available": False, "reason": "CPU"}]

        async def reset_prefix_cache(self):
            return self.base.reset_prefix_cache()

        def shutdown(self, timeout):
            self.base.shutdown()

    async def create(self, kwargs):
        return Native(kwargs)

    monkeypatch.setattr(AsyncEngineFacade, "_create", create)
    return case


def execute(case, output):
    return verification._verify_with_profile(
        case["recipe"], output, build_route(case["recipe"])
    )


def test_complete_serving_route_binds_streams_and_separates_reports(
    serving_case, tmp_path
):
    recipe = serving_case["recipe"]
    serving_case["recipe"] = replace(
        recipe,
        policy=VerificationPolicy(
            (
                PolicyCriterion("quality.candidate_pass_rate", "1", minimum=1),
                PolicyCriterion("serving.ttft_seconds_candidate", "1", maximum=60),
            )
        ),
    )
    result = execute(serving_case, tmp_path / "result")
    assert result.manifest["verdict"] == "PASS", result.to_data()
    assert all(engine.closed for engine in serving_case["engines"])
    report = (result.output_dir / "report.md").read_text()
    assert "private" not in report
    assert "Client concurrency: 1 -> 2" in report
    assert "ttft_seconds:" in report and "allocated_kv_bytes: unavailable" in report
    data = result.to_data()
    assert data["performance"]["metrics"]["output_tokens_per_second"]["available"]
    assert "ttft_seconds" not in data["performance"]["unsupported_metrics"]
    assert "allocated_kv_bytes" in data["performance"]["unsupported_metrics"]
    left = load_run_record(result.output_dir / "reference.run.json")
    right = load_run_record(result.output_dir / "candidate.run.json")
    assert not evidence_gaps(
        right,
        runtime_pin="a" * 64,
        config=recipe.measurement_configs[TokenTrajectoryProtocol.name],
    )
    assert compare_serving(left, right, False)["available"] is False


def test_missing_cuda_evidence_cannot_pass_memory_policy(serving_case, tmp_path):
    serving_case["recipe"] = replace(
        serving_case["recipe"],
        policy=VerificationPolicy(
            (
                PolicyCriterion(
                    "serving.allocated_kv_bytes_candidate", "1", maximum=10**12
                ),
            )
        ),
    )
    result = execute(serving_case, tmp_path / "cpu")
    assert result.manifest["verdict"] == "INSUFFICIENT_EVIDENCE"


@pytest.mark.parametrize("mode", ["capacity", "reset_failed"])
def test_native_controls_must_be_observed(serving_case, tmp_path, mode):
    serving_case["mode"] = mode
    result = execute(serving_case, tmp_path / mode)
    assert result.exit_code == 5


@pytest.mark.parametrize(
    "field,value",
    [("max_num_seqs", 3), ("enable_prefix_caching", True), ("dtype", "float16")],
)
def test_unrelated_changes_are_rejected_before_execution(serving_case, field, value):
    recipe = serving_case["recipe"]
    left, right = recipe.study.runs
    bad = replace(
        recipe,
        study=replace(
            recipe.study,
            runs=(left, replace(right, runtime={**right.runtime, field: value})),
        ),
    )
    with pytest.raises(ValueError):
        build_route(bad)
    assert not serving_case["engines"]


def test_preparation_holds_capacity_and_model_pins_fixed(serving_case, monkeypatch):
    from metria import preparation

    base = serving_case["recipe"]
    monkeypatch.setattr(
        preparation,
        "installed_runtime_identity",
        lambda: {"version": "0.30.0+cpu", "sha256": "a" * 64},
    )
    model = base.study.runs[0].model
    recipe = prepare_serving_recipe(
        model["path"],
        {"files": model["files"]},
        base.measurement_configs[TokenTrajectoryProtocol.name]["prompts"],
    )
    assert recipe.study.comparison.vary == SERVING_VARIATIONS
    assert [run.trial_policy["concurrency"] for run in recipe.study.runs] == [1, 2]


@pytest.mark.parametrize(
    "mutation",
    ["metrics", "trials", "concurrency", "tokens", "prompts", "reset", "method"],
)
def test_tampered_measurement_evidence_cannot_produce_valid_serving_comparison(
    serving_case, tmp_path, mutation
):
    result = execute(serving_case, tmp_path / mutation)
    record = load_run_record(result.output_dir / "candidate.run.json")
    from metria.recipes import _json_value

    data = _json_value(record.evidence, path="evidence")
    capture = data["measurements"][TokenTrajectoryProtocol.name]
    workload = capture["workload"]
    if mutation == "metrics":
        metrics = dict(record.metrics)
        metric = metrics["ttft_seconds"]
        metrics["ttft_seconds"] = replace(metric, value=999)
        record = replace(record, metrics=metrics)
    elif mutation == "trials":
        workload["batches"].pop()
    elif mutation == "concurrency":
        workload["batches"][0]["peak_inflight"] = 1
    elif mutation == "tokens":
        workload["batches"][0]["requests"][0]["output_tokens"] = 99
    elif mutation == "prompts":
        capture["prompts"][0]["prompt_sha256"] = "b" * 64
    elif mutation == "method":
        workload["method"] = "other"
    else:
        record = replace(record, observed={**record.observed, "reset_events": ()})
    record = replace(record, evidence=data)
    assert evidence_gaps(
        record,
        runtime_pin="a" * 64,
        config=serving_case["recipe"].measurement_configs[TokenTrajectoryProtocol.name],
    )


def test_installed_command_prepares_recipe_and_refuses_overwrite(
    serving_case, monkeypatch, tmp_path
):
    import io
    import json

    from metria import cli, onboarding, preparation

    monkeypatch.setattr(onboarding, "_runtime_host_supported", lambda: True)
    monkeypatch.setattr(
        preparation,
        "installed_runtime_identity",
        lambda: {"version": "0.30.0+cpu", "sha256": "a" * 64},
    )
    recipe = serving_case["recipe"]
    model = recipe.study.runs[0].model
    descriptor, workload, policy, output = (
        tmp_path / name
        for name in ("descriptor.json", "workload.jsonl", "policy.json", "study.json")
    )
    descriptor.write_text(json.dumps({"files": dict(model["files"])}))
    workload.write_text(
        "\n".join(
            json.dumps({"id": str(i), "prompt": "public example"}) for i in range(2)
        )
    )
    policy.write_text(
        json.dumps(
            VerificationPolicy(
                (PolicyCriterion("serving.ttft_seconds_candidate", "1", maximum=1),)
            ).to_data()
        )
    )
    args = [
        "recipe",
        "prepare-vllm-serving",
        "--model",
        model["path"],
        "--descriptor",
        str(descriptor),
        "--workload",
        str(workload),
        "--policy",
        str(policy),
        "--output",
        str(output),
    ]
    stdout, stderr = io.StringIO(), io.StringIO()
    assert cli.main(args, stdout=stdout, stderr=stderr) == 0, stderr.getvalue()
    original = output.read_bytes()
    assert cli.main(args, stdout=stdout, stderr=stderr) == 2
    assert output.read_bytes() == original
    args[-1] = str(tmp_path / "new.json")
    descriptor.write_text("[]")
    assert cli.main(args, stdout=stdout, stderr=stderr) == 2
    monkeypatch.setattr(onboarding, "_runtime_host_supported", lambda: False)
    assert cli.main(args, stdout=stdout, stderr=stderr) == 2


def test_serving_measurements_reject_mismatched_identity_and_workloads(
    serving_case, tmp_path
):
    from metria.measurements.serving_impact import serving_result
    from metria.models import RunStatus
    from metria.recipes import _json_value

    result = execute(serving_case, tmp_path / "compare")
    left = load_run_record(result.output_dir / "reference.run.json")
    right = load_run_record(result.output_dir / "candidate.run.json")
    assert (
        serving_result(replace(left, status=RunStatus.FAILED)).evidence["available"]
        is False
    )
    metrics = dict(right.metrics)
    metrics["ttft_seconds"] = replace(
        metrics["ttft_seconds"],
        definition=replace(metrics["ttft_seconds"].definition, unit="milliseconds"),
    )
    comparison = compare_serving(left, replace(right, metrics=metrics), True)
    assert comparison["metrics"]["ttft_seconds"]["available"] is False
    assert comparison["metrics"]["requests_per_second"]["available"] is True
    for field, value in (
        ("method", "other"),
        ("measured_trials", 0),
        ("measured_trials", 3),
        ("batches", []),
    ):
        evidence = _json_value(right.evidence, path="evidence")
        evidence["measurements"][TokenTrajectoryProtocol.name]["workload"][field] = (
            value
        )
        assert (
            serving_result(replace(right, evidence=evidence)).evidence["available"]
            is False
        )
    changed = replace(
        right,
        requested=replace(
            right.requested, scenario={**right.requested.scenario, "seed": 12}
        ),
    )
    assert all(
        not row["available"]
        for row in compare_serving(left, changed, True)["metrics"].values()
    )


@pytest.mark.parametrize("mode", ["cpu", "gpu"])
def test_retained_native_serving_evidence_reproduces_current_metrics(mode):
    import hashlib
    import json
    from pathlib import Path

    from metria.recipes import load_study_recipe

    root = Path(__file__).parents[1] / "artifacts/qualification/vllm-serving"
    hashes = json.loads((root / "sha256.json").read_text())
    for name, digest in hashes.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    case = root / mode
    recipe = load_study_recipe(case / "study.json")
    records = [
        load_run_record(case / f"verification/{role}.run.json")
        for role in ("reference", "candidate")
    ]
    for record in records:
        assert not evidence_gaps(
            record,
            runtime_pin=recipe.environment["vllm_distribution_sha256"],
            config=recipe.measurement_configs[TokenTrajectoryProtocol.name],
        )
    current = compare_serving(*records, True)
    projected = json.loads((case / "current-projection.json").read_text())
    assert current == projected["performance"]
    assert current["metrics"]["allocated_kv_bytes"]["available"] is (mode == "gpu")
    assert current["metrics"]["ttft_seconds"]["available"] is True
    assert "ttft_seconds" not in current["unsupported_metrics"]
