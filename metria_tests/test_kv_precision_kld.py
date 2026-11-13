"""STAGED trajectory KL divergence inside FP8 KV-cache precision verification."""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
from metria_tests.test_kv_precision_verify import _execute, kv_case  # noqa: F401
from metria_tests.test_vllm_verify import prefix_case  # noqa: F401

from metria.measurements import (
    TokenTrajectoryProtocol,
    TrajectoryAgreementAnalysis,
    TrajectoryDivergenceAnalysis,
)
from metria.measurements.verification_impact import VerificationImpactAnalysis
from metria.policies import PolicyCriterion, VerificationPolicy, policy_from_data
from metria.records import load_run_record
from metria.runtimes import vllm
from metria.verification_kv_precision import validate_kv_precision_recipe
from metria.verification_vllm import trial_identity, validate_vllm_recipe

ANALYSES = (
    TrajectoryAgreementAnalysis.name,
    VerificationImpactAnalysis.name,
    TrajectoryDivergenceAnalysis.name,
)


def _policy(maximum: float) -> VerificationPolicy:
    return VerificationPolicy(
        (
            PolicyCriterion("quality.candidate_pass_rate", "1", minimum=1.0),
            PolicyCriterion("behavior.trajectory_kld_mean", "0.1.0", maximum=maximum),
        )
    )


def _steps(shift: float) -> dict[int, SimpleNamespace]:
    raw = [math.exp(-0.4 * rank) for rank in range(20)]
    raw[0] *= math.exp(shift)
    norm = sum(raw) / 0.95
    return {
        100 + rank: SimpleNamespace(logprob=math.log(value / norm), rank=rank + 1)
        for rank, value in enumerate(raw)
    }


def _logprob_engine(monkeypatch, *, mode: str = "raw_logprobs") -> None:
    module = vllm._load_vllm()
    base = module.LLM

    class Engine(base):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.model_config.logprobs_mode = mode
            # FP8 moves the candidate distributions; the reference stays put.
            self.shift = 0.3 if kwargs["kv_cache_dtype"] != "auto" else 0.0

        def generate(self, prompts, **kwargs):
            responses = super().generate(prompts, **kwargs)
            params = kwargs["sampling_params"]
            for response, param in zip(responses, params, strict=True):
                if getattr(param, "logprobs", None) == 20:
                    output = response.outputs[0]
                    output.logprobs = [_steps(self.shift) for _ in output.token_ids]
            return responses

    patched = type(module)(**{**vars(module), "LLM": Engine})
    monkeypatch.setattr(vllm, "_load_vllm", lambda: patched)


def _staged(case, *, flag=True, analyses=ANALYSES, policy=None):
    recipe = case["recipe"]
    config = {
        **recipe.measurement_configs[TokenTrajectoryProtocol.name],
        "staged_distributions": flag,
    }
    runs = tuple(
        replace(run, trial_policy=trial_identity(config)) for run in recipe.study.runs
    )
    case["recipe"] = replace(
        recipe,
        study=replace(
            recipe.study,
            runs=runs,
            comparison=replace(recipe.study.comparison, analyses=analyses),
        ),
        measurement_configs={TokenTrajectoryProtocol.name: config},
        policy=policy if policy is not None else _policy(1.0),
    )
    return case


def test_staged_kld_policy_passes_and_is_labeled(kv_case, monkeypatch, tmp_path):  # noqa: F811
    _logprob_engine(monkeypatch)
    result = _execute(_staged(kv_case), tmp_path / "kld")
    manifest = result.manifest
    assert manifest["verdict"] == "PASS", manifest
    rows = {row["criterion"]["target"]: row for row in manifest["policy"]["criteria"]}
    kld = rows["behavior.trajectory_kld_mean"]
    assert kld["result"] == "PASS"
    assert kld["target_identity"]["qualification"] == "staged"
    assert 0.0 < kld["observed_value"] < 1.0
    assert (
        "behavior.trajectory_kld_mean@0.1.0 [staged]"
        in (result.output_dir / "report.md").read_text()
    )
    candidate = load_run_record(result.output_dir / "candidate.run.json")
    capture = candidate.evidence["measurements"][TokenTrajectoryProtocol.name]
    retained = capture["distributions"]
    assert retained["qualification"] == "staged"
    assert retained["n_prompts"] == capture["n_prompts"] == 4
    assert [len(step) for step in retained["prompts"][0]["top_logprobs"]] == [20] * 3
    assert list(candidate.requested.trial_policy["capability_overrides"]) == [
        "metria.staged_captures"
    ]
    assert all(
        row["top_logprobs"]["mode"] == "raw_logprobs"
        for row in candidate.observed["invocations"]
    )


def test_staged_kld_policy_can_fail_the_change(kv_case, monkeypatch, tmp_path):  # noqa: F811
    _logprob_engine(monkeypatch)
    result = _execute(_staged(kv_case, policy=_policy(1e-6)), tmp_path / "tight")
    assert result.manifest["verdict"] == "FAIL"


def test_processed_logprobs_cannot_reach_a_policy_decision(
    kv_case,  # noqa: F811
    monkeypatch,
    tmp_path,
):
    _logprob_engine(monkeypatch, mode="processed_logprobs")
    result = _execute(_staged(kv_case), tmp_path / "processed")
    assert result.manifest["verdict"] not in {"PASS", "FAIL", "VERIFIED"}


@pytest.mark.parametrize(
    ("flag", "analyses", "match"),
    [
        (True, ANALYSES[:2], "exactly when staged_distributions"),
        (False, ANALYSES, "exactly when staged_distributions"),
    ],
)
def test_kld_analysis_and_flag_must_agree(kv_case, flag, analyses, match):  # noqa: F811
    policy = VerificationPolicy(
        (PolicyCriterion("quality.candidate_pass_rate", "1", minimum=1.0),)
    )
    case = _staged(kv_case, flag=flag, analyses=analyses, policy=policy)
    with pytest.raises(ValueError, match=match):
        validate_kv_precision_recipe(case["recipe"])


def test_kld_policy_requires_the_declared_analysis(kv_case):  # noqa: F811
    with pytest.raises(ValueError, match="undeclared analyses"):
        _staged(kv_case, flag=False, analyses=ANALYSES[:2])


def test_other_vllm_profiles_reject_staged_distributions(prefix_case):  # noqa: F811
    recipe = prefix_case["recipe"]
    config = {
        **recipe.measurement_configs[TokenTrajectoryProtocol.name],
        "staged_distributions": True,
    }
    recipe = replace(
        recipe,
        study=replace(
            recipe.study,
            runs=tuple(
                replace(run, trial_policy=trial_identity(config))
                for run in recipe.study.runs
            ),
        ),
        measurement_configs={TokenTrajectoryProtocol.name: config},
    )
    with pytest.raises(ValueError, match="only by KV-precision"):
        validate_vllm_recipe(recipe)


def test_prepared_staged_kld_recipe(kv_case, monkeypatch):  # noqa: F811
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
        kv_cache_dtype="fp8",
        staged_distributions=True,
    )
    assert recipe.study.comparison.analyses == ANALYSES
    import json

    from metria.recipes import study_recipe_from_data, study_recipe_to_json

    loaded = study_recipe_from_data(
        json.loads(study_recipe_to_json(replace(recipe, policy=_policy(0.5))))
    )
    assert validate_kv_precision_recipe(loaded) == ("auto", "fp8")
    assert validate_kv_precision_recipe(replace(recipe, policy=_policy(0.5))) == (
        "auto",
        "fp8",
    )
    with pytest.raises(ValueError, match="KV-cache precision recipe"):
        preparation.prepare_vllm_prefix_recipe(
            run.model["path"],
            {"files": run.model["files"]},
            prompts,
            staged_distributions=True,
        )


def test_kld_targets_parse_with_nats_and_reject_negative_bounds() -> None:
    policy = policy_from_data(
        {
            "schema": "metria.verification_policy.v1",
            "criteria": [
                {
                    "target": "behavior.trajectory_kld_max",
                    "version": "0.1.0",
                    "unit": "nats",
                    "max": 2.5,
                }
            ],
        }
    )
    assert policy.required_analyses == {"metria.trajectory_kld"}
    with pytest.raises(ValueError, match="target domain"):
        PolicyCriterion("behavior.trajectory_kld_mean", "0.1.0", maximum=-1.0)
    with pytest.raises(ValueError, match="unknown policy target"):
        PolicyCriterion("behavior.trajectory_kld_mean", "0.3.4", maximum=1.0)
