"""STAGED local_vllm_parallelism.v1 with a fake multi-device vLLM engine."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from metria_tests.test_vllm_verify import prefix_case  # noqa: F401

from metria import verification
from metria.measurements import TokenTrajectoryProtocol, TrajectoryAgreementAnalysis
from metria.measurements.verification_impact import VerificationImpactAnalysis
from metria.policies import PolicyCriterion, VerificationPolicy
from metria.runtimes import vllm
from metria.verification_parallelism import (
    build_route,
    parallel_vary,
    validate_parallelism_recipe,
)

POLICY = VerificationPolicy(
    (
        PolicyCriterion("quality.candidate_pass_rate", "1", minimum=1.0),
        PolicyCriterion("behavior.trajectory_agreement", "0.3.4", minimum=0.5),
    )
)


def _hardware(devices: int):
    return lambda: {
        "status": "observed",
        "device_type": "cuda",
        "source": "fixture",
        "devices": [{"logical_device": i, "name": "L4"} for i in range(devices)],
    }


def _patch_engine(monkeypatch, *, ignore_parallel: bool = False) -> None:
    module = vllm._load_vllm()
    base = module.LLM

    class Engine(base):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.llm_engine.vllm_config.parallel_config = SimpleNamespace(
                tensor_parallel_size=1
                if ignore_parallel
                else kwargs["tensor_parallel_size"],
                pipeline_parallel_size=kwargs.get("pipeline_parallel_size", 1),
            )

    patched = type(module)(**{**vars(module), "LLM": Engine})
    monkeypatch.setattr(vllm, "_load_vllm", lambda: patched)


@pytest.fixture
def parallel_case(prefix_case, monkeypatch):  # noqa: F811
    monkeypatch.setattr(vllm, "_vllm_version", lambda: "0.30.0+cu129")
    monkeypatch.setattr(vllm, "native_hardware", _hardware(2))
    monkeypatch.setattr(
        vllm,
        "require_runtime_pin",
        lambda *a, **kw: {
            "status": "verified",
            "sha256": "a" * 64,
            "version": "0.30.0+cu129",
        },
    )
    _patch_engine(monkeypatch)
    recipe = prefix_case["recipe"]
    config = dict(recipe.measurement_configs[TokenTrajectoryProtocol.name])
    config["prompts"] = [
        {**row, "checks": [{"id": "answers", "kind": "nonempty"}]}
        for row in config["prompts"]
    ]
    prefix_case["config"] = config
    prefix_case["recipe"] = _recipe(recipe, config, {"tensor_parallel_size": 2})
    return prefix_case


def _recipe(base, config, candidate, reference=None):
    reference = reference or {key: 1 for key in candidate}
    runs = tuple(
        replace(
            run,
            runtime={
                **run.runtime,
                "version": "0.30.0+cu129",
                "enable_prefix_caching": False,
                **sizes,
            },
        )
        for run, sizes in zip(base.study.runs, (reference, candidate), strict=True)
    )
    return replace(
        base,
        study=replace(
            base.study,
            runs=runs,
            comparison=replace(
                base.study.comparison,
                vary=parallel_vary(runs[0], runs[1]),
                analyses=(
                    TrajectoryAgreementAnalysis.name,
                    VerificationImpactAnalysis.name,
                ),
            ),
        ),
        measurement_configs={TokenTrajectoryProtocol.name: config},
        policy=POLICY,
    )


def _execute(recipe, output):
    return verification._verify_with_profile(recipe, output, build_route(recipe))


def test_tensor_parallel_candidate_reaches_a_policy_decision(parallel_case, tmp_path):
    result = _execute(parallel_case["recipe"], tmp_path / "tp")
    manifest = result.manifest
    assert manifest["verdict"] == "PASS", manifest["records"]
    assert manifest["scope"] == "local_vllm_parallelism.v1"
    assert manifest["change"]["candidate"] == {
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 1,
    }
    facts = manifest["records"]["candidate"]["observed"]
    assert facts["tensor_parallel_size"] == 2 and facts["device_count"] == 2
    report = (result.output_dir / "report.md").read_text()
    assert "STAGED local vLLM tensor/pipeline parallelism" in report
    assert "Observed tensor parallel: 2" in report


def test_pipeline_parallel_candidate_is_read_back(parallel_case, tmp_path):
    recipe = _recipe(
        parallel_case["recipe"],
        parallel_case["config"],
        {"tensor_parallel_size": 1, "pipeline_parallel_size": 2},
    )
    manifest = _execute(recipe, tmp_path / "pp").manifest
    assert manifest["verdict"] == "PASS", manifest["records"]
    facts = manifest["records"]["candidate"]["observed"]
    assert facts["pipeline_parallel_size"] == 2


def test_engine_ignoring_parallelism_cannot_pass(parallel_case, monkeypatch, tmp_path):
    _patch_engine(monkeypatch, ignore_parallel=True)
    manifest = _execute(parallel_case["recipe"], tmp_path / "x").manifest
    assert manifest["verdict"] in {"EXECUTION_FAILED", "INSUFFICIENT_EVIDENCE"}


def test_too_few_devices_is_insufficient_evidence(parallel_case, monkeypatch, tmp_path):
    monkeypatch.setattr(vllm, "native_hardware", _hardware(1))
    manifest = _execute(parallel_case["recipe"], tmp_path / "x").manifest
    assert manifest["verdict"] == "INSUFFICIENT_EVIDENCE"
    gaps = manifest["records"]["candidate"]["evidence_gaps"]
    assert any("enough CUDA devices" in gap for gap in gaps)


def test_dispatch_routes_parallel_recipes(parallel_case):
    route = verification._verification_route(parallel_case["recipe"])
    assert route.scope == "local_vllm_parallelism.v1"


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ("reference_parallel", "single-device reference"),
        ("too_many", "single-device reference"),
        ("no_policy", "explicit policy"),
        ("wrong_vary", "varies only"),
        ("cpu_wheel", "qualified vLLM"),
    ],
)
def test_invalid_parallel_recipes_are_rejected(parallel_case, change, match):
    recipe, config = parallel_case["recipe"], parallel_case["config"]
    if change == "reference_parallel":
        recipe = _recipe(
            recipe, config, {"tensor_parallel_size": 2}, {"tensor_parallel_size": 2}
        )
    elif change == "too_many":
        recipe = _recipe(recipe, config, {"tensor_parallel_size": 16})
    elif change == "no_policy":
        recipe = replace(recipe, policy=None)
    elif change == "wrong_vary":
        plan = replace(recipe.study.comparison, vary=frozenset({"runtime.dtype"}))
        recipe = replace(recipe, study=replace(recipe.study, comparison=plan))
    else:
        runs = tuple(
            replace(run, runtime={**run.runtime, "version": "0.30.0+cpu"})
            for run in recipe.study.runs
        )
        recipe = replace(recipe, study=replace(recipe.study, runs=runs))
    with pytest.raises(ValueError, match=match):
        validate_parallelism_recipe(recipe)


def test_prepared_parallel_recipe_validates_with_a_policy(parallel_case, monkeypatch):
    from metria import preparation

    run = parallel_case["recipe"].study.runs[0]
    monkeypatch.setattr(
        preparation,
        "installed_runtime_identity",
        lambda: {"version": "0.30.0+cu129", "sha256": "a" * 64},
    )
    recipe = preparation.prepare_vllm_prefix_recipe(
        run.model["path"],
        {"files": run.model["files"]},
        parallel_case["config"]["prompts"],
        candidate_parallel={"tensor_parallel_size": 2},
    )
    assert [r.runtime["tensor_parallel_size"] for r in recipe.study.runs] == [1, 2]
    with pytest.raises(ValueError, match="explicit policy"):
        validate_parallelism_recipe(recipe)
    sizes = validate_parallelism_recipe(replace(recipe, policy=POLICY))
    assert sizes[1]["tensor_parallel_size"] == 2
    with pytest.raises(ValueError, match="not both"):
        preparation.prepare_vllm_prefix_recipe(
            run.model["path"],
            {"files": run.model["files"]},
            parallel_case["config"]["prompts"],
            kv_cache_dtype="fp8",
            candidate_parallel={"tensor_parallel_size": 2},
        )


def test_cuda_devices_lists_every_visible_device():
    from metria.runtimes.vllm_hardware import cuda_devices

    def properties(index):
        return SimpleNamespace(
            name=f"GPU{index}", major=8, minor=9, total_memory=24 << 30, uuid=index
        )

    torch = SimpleNamespace(
        cuda=SimpleNamespace(device_count=lambda: 2, get_device_properties=properties)
    )
    devices = cuda_devices(torch)
    assert [row["name"] for row in devices] == ["GPU0", "GPU1"]
    assert devices[1]["compute_capability"] == [8, 9]
    assert len(devices[0]["uuid_sha256"]) == 64
