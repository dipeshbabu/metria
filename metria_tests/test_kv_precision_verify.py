"""Qualified vLLM KV-cache precision verification (automatic vs FP8 cache)."""

from __future__ import annotations

from dataclasses import replace

import pytest
from metria_tests.test_vllm_verify import prefix_case  # noqa: F401

from metria import TreatmentSpec, TreatmentType, verification
from metria.measurements import TokenTrajectoryProtocol, TrajectoryAgreementAnalysis
from metria.measurements.verification_impact import VerificationImpactAnalysis
from metria.policies import PolicyCriterion, VerificationPolicy
from metria.records import load_run_record
from metria.runtimes import vllm
from metria.verification_kv_precision import (
    KV_PRECISION_VARIATIONS,
    build_route,
    validate_kv_precision_recipe,
)
from metria.verification_schema import VLLM_KV_PRECISION_SCOPE

POLICY = VerificationPolicy(
    (
        PolicyCriterion("quality.candidate_pass_rate", "1", minimum=1.0),
        PolicyCriterion("behavior.trajectory_agreement", "0.3.4", minimum=0.5),
    )
)


def _treatment(dtype: str) -> TreatmentSpec:
    return TreatmentSpec(
        name="vllm.kv_cache",
        kind=TreatmentType.RUNTIME_FEATURE,
        config={"dtype": dtype},
    )


@pytest.fixture
def kv_case(prefix_case, monkeypatch):  # noqa: F811
    monkeypatch.setattr(vllm, "_vllm_version", lambda: "0.30.0+cu129")
    monkeypatch.setattr(
        vllm,
        "native_hardware",
        lambda: {"status": "observed", "device_type": "cuda", "source": "fixture"},
    )
    monkeypatch.setattr(
        vllm,
        "require_runtime_pin",
        lambda *a, **kw: {
            "status": "verified",
            "sha256": "a" * 64,
            "version": "0.30.0+cu129",
        },
    )
    recipe = prefix_case["recipe"]
    config = recipe.measurement_configs[TokenTrajectoryProtocol.name]
    config = {
        **config,
        "prompts": [
            {**row, "checks": [{"id": "answers", "kind": "nonempty"}]}
            for row in config["prompts"]
        ],
    }
    runs = tuple(
        replace(
            run,
            runtime={
                **run.runtime,
                "version": "0.30.0+cu129",
                "enable_prefix_caching": False,
            },
            treatments=(_treatment(dtype),),
        )
        for run, dtype in zip(recipe.study.runs, ("auto", "fp8"), strict=True)
    )
    prefix_case["recipe"] = replace(
        recipe,
        study=replace(
            recipe.study,
            runs=runs,
            comparison=replace(
                recipe.study.comparison,
                vary=KV_PRECISION_VARIATIONS,
                analyses=(
                    TrajectoryAgreementAnalysis.name,
                    VerificationImpactAnalysis.name,
                ),
            ),
        ),
        measurement_configs={TokenTrajectoryProtocol.name: config},
        policy=POLICY,
    )
    return prefix_case


def _execute(case, output):
    return verification._verify_with_profile(
        case["recipe"], output, build_route(case["recipe"])
    )


def test_fp8_cache_verification_reaches_a_policy_decision(kv_case, tmp_path):
    result = _execute(kv_case, tmp_path / "kv")
    assert result.exit_code == 0, result.to_data()
    manifest = result.manifest
    assert manifest["scope"] == VLLM_KV_PRECISION_SCOPE
    assert manifest["verdict"] == "PASS"
    assert manifest["change"] == {
        "parameter": "kv_cache_dtype",
        "reference": "auto",
        "candidate": "fp8",
    }
    assert manifest["records"]["candidate"]["observed"]["kv_cache_dtype"] == "fp8"
    report = (result.output_dir / "report.md").read_text()
    assert "KV-cache dtype: auto -> fp8" in report
    assert "Observed KV-cache dtype: fp8" in report
    candidate = load_run_record(result.output_dir / "candidate.run.json")
    assert candidate.resolved["kv_cache"]["dtype"] == "fp8"


def _patch_engine(monkeypatch, adjust):
    module = vllm._load_vllm()
    base = module.LLM

    class Engine(base):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            adjust(self.llm_engine.vllm_config.cache_config)

    patched = type(module)(**{**vars(module), "LLM": Engine})
    monkeypatch.setattr(vllm, "_load_vllm", lambda: patched)


def test_engine_contradicting_fp8_fails_before_measurement(
    kv_case, tmp_path, monkeypatch
):
    _patch_engine(monkeypatch, lambda cache: setattr(cache, "cache_dtype", "auto"))
    result = _execute(kv_case, tmp_path / "contradicted")
    assert result.exit_code == 5
    assert result.manifest["verdict"] == "EXECUTION_FAILED"
    assert result.manifest["policy_status"] == "NOT_EVALUATED"


def test_engine_without_cache_dtype_readback_is_insufficient_evidence(
    kv_case, tmp_path, monkeypatch
):
    _patch_engine(monkeypatch, lambda cache: delattr(cache, "cache_dtype"))
    result = _execute(kv_case, tmp_path / "unobserved")
    assert result.exit_code == 4, result.manifest["verdict"]
    gaps = result.manifest["records"]["candidate"]["evidence_gaps"]
    assert "runtime-applied KV-cache dtype does not match the request" in gaps


def test_route_dispatch_selects_the_kv_precision_profile(kv_case):
    route = verification._verification_route(kv_case["recipe"])
    assert route.scope == VLLM_KV_PRECISION_SCOPE
    assert route.isolated is True


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ("no_policy", "acceptance policy"),
        ("no_checks", "task checks"),
        ("auto_candidate", "FP8 candidate"),
        ("cpu_wheel", "qualified vLLM"),
        ("prefix_on", "prefix caching off"),
        ("extra_treatment", "exactly one vllm.kv_cache"),
        ("wrong_vary", "five KV-cache variation"),
    ],
)
def test_invalid_kv_precision_recipes_are_rejected(kv_case, change, match):
    recipe = kv_case["recipe"]
    left, right = recipe.study.runs
    if change == "no_policy":
        recipe = replace(recipe, policy=None)
    elif change == "no_checks":
        config = recipe.measurement_configs[TokenTrajectoryProtocol.name]
        config = {
            **config,
            "prompts": [
                {k: v for k, v in row.items() if k != "checks"}
                for row in config["prompts"]
            ],
        }
        recipe = replace(
            recipe, measurement_configs={TokenTrajectoryProtocol.name: config}
        )
    elif change == "auto_candidate":
        recipe = replace(
            recipe,
            study=replace(
                recipe.study,
                runs=(left, replace(right, treatments=(_treatment("auto"),))),
            ),
        )
    elif change == "cpu_wheel":
        runs = tuple(
            replace(run, runtime={**run.runtime, "version": "0.30.0+cpu"})
            for run in (left, right)
        )
        recipe = replace(recipe, study=replace(recipe.study, runs=runs))
    elif change == "prefix_on":
        runs = tuple(
            replace(run, runtime={**run.runtime, "enable_prefix_caching": True})
            for run in (left, right)
        )
        recipe = replace(recipe, study=replace(recipe.study, runs=runs))
    elif change == "extra_treatment":
        staged = TreatmentSpec(
            name="vllm.cache_layout",
            kind=TreatmentType.RUNTIME_FEATURE,
            config={"block_size": 32},
        )
        recipe = replace(
            recipe,
            study=replace(
                recipe.study,
                runs=(left, replace(right, treatments=(_treatment("fp8"), staged))),
            ),
        )
    else:
        recipe = replace(
            recipe,
            study=replace(
                recipe.study,
                comparison=replace(
                    recipe.study.comparison, vary=frozenset({"treatments"})
                ),
            ),
        )
    with pytest.raises((ValueError, TypeError), match=match):
        validate_kv_precision_recipe(recipe)


def test_retained_l4_records_satisfy_the_kv_readback_contract():
    from metria.verification_kv_precision import _kv_gaps

    base = "artifacts/qualification/l4-native-gpu/fp8-kv-cache/"
    for name in ("reference", "fp8"):
        record = load_run_record(base + f"{name}.run.json")
        assert _kv_gaps(record) == []


def test_prepared_fp8_recipe_validates_once_a_policy_is_set(kv_case, monkeypatch):
    from metria import preparation

    run = kv_case["recipe"].study.runs[0]
    prompts = kv_case["recipe"].measurement_configs[TokenTrajectoryProtocol.name][
        "prompts"
    ]
    monkeypatch.setattr(
        preparation,
        "installed_runtime_identity",
        lambda: {"version": "0.30.0+cu129", "sha256": "a" * 64},
    )
    recipe = preparation.prepare_vllm_prefix_recipe(
        run.model["path"],
        {"files": run.model["files"]},
        prompts,
        kv_cache_dtype="fp8_e4m3",
    )
    assert [r.treatments[0].config["dtype"] for r in recipe.study.runs] == [
        "auto",
        "fp8_e4m3",
    ]
    assert all(r.runtime["enable_prefix_caching"] is False for r in recipe.study.runs)
    with pytest.raises(ValueError, match="acceptance policy"):
        validate_kv_precision_recipe(recipe)
    assert validate_kv_precision_recipe(replace(recipe, policy=POLICY)) == (
        "auto",
        "fp8_e4m3",
    )
    with pytest.raises(ValueError, match="kv_cache_dtype"):
        preparation.prepare_vllm_prefix_recipe(
            run.model["path"],
            {"files": run.model["files"]},
            prompts,
            kv_cache_dtype="int8",
        )
