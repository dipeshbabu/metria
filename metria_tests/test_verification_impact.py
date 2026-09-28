from dataclasses import replace
from itertools import count

import pytest
from metria_tests.test_vllm_verify import prefix_case as _prefix_case_fixture

from metria import PolicyCriterion, VerificationPolicy
from metria.measurements import (
    TokenTrajectoryProtocol,
    TrajectoryAgreementAnalysis,
    prefix_workload,
)
from metria.measurements.verification_impact import (
    VerificationImpactAnalysis,
    reference_variability,
)
from metria.records import load_run_record
from metria.verification import _verify_with_profile
from metria.verification_vllm import build_route

prefix_case = _prefix_case_fixture


def decision_recipe(case, checks, criteria):
    recipe = case["recipe"]
    config = recipe.measurement_configs[TokenTrajectoryProtocol.name]
    rows = [{**row, "checks": checks} for row in config["prompts"]]
    return replace(
        recipe,
        study=replace(
            recipe.study,
            comparison=replace(
                recipe.study.comparison,
                analyses=(
                    TrajectoryAgreementAnalysis.name,
                    VerificationImpactAnalysis.name,
                ),
            ),
        ),
        measurement_configs={TokenTrajectoryProtocol.name: {**config, "prompts": rows}},
        policy=VerificationPolicy(tuple(criteria)),
    )


def run_recipe(recipe, output):
    return _verify_with_profile(recipe, output, build_route(recipe))


def test_quality_repeatability_and_latency_have_separate_policy_targets(
    prefix_case, tmp_path, monkeypatch
):
    ticks = count(0, 1000)
    monkeypatch.setattr(prefix_workload.time, "perf_counter_ns", lambda: next(ticks))
    recipe = decision_recipe(
        prefix_case,
        [{"id": "answer", "kind": "exact_text", "expected": "private completion"}],
        [
            PolicyCriterion("quality.candidate_pass_rate", "1", minimum=1),
            PolicyCriterion("behavior.reference_repeatability", "1", minimum=1),
            PolicyCriterion("performance.latency_ratio", "1", maximum=1.1),
        ],
    )
    result = run_recipe(recipe, tmp_path / "pass")
    assert result.exit_code == 0 and result.manifest["verdict"] == "PASS"
    impact = result.manifest["analyses"][1]
    assert impact["diagnostics"]["quality"]["candidate_passed"] == 4
    assert impact["diagnostics"]["reference_variability"]["different_pairs"] == 0
    assert impact["metrics"]["latency_ratio"]["value"] == 1
    assert (
        "private completion"
        not in (result.output_dir / "verification.json").read_text()
    )


def test_exact_trajectory_agreement_does_not_hide_failed_task_checks(
    prefix_case, tmp_path
):
    recipe = decision_recipe(
        prefix_case,
        [
            {
                "id": "answer",
                "kind": "exact_text",
                "expected": "different expected answer",
            }
        ],
        [PolicyCriterion("quality.candidate_pass_rate", "1", minimum=1)],
    )
    result = run_recipe(recipe, tmp_path / "fail")
    assert result.exit_code == 1 and result.manifest["verdict"] == "FAIL"
    assert (
        result.manifest["analyses"][0]["metrics"]["trajectory_agreement_score"]["value"]
        == 100
    )
    assert (
        result.manifest["analyses"][1]["metrics"]["candidate_task_pass_rate"]["value"]
        == 0
    )


def test_unconfigured_quality_cannot_pass_even_a_zero_minimum(prefix_case, tmp_path):
    recipe = decision_recipe(
        prefix_case,
        [],
        [PolicyCriterion("quality.candidate_pass_rate", "1", minimum=0)],
    )
    result = run_recipe(recipe, tmp_path / "missing")
    assert result.exit_code == 4
    assert result.manifest["policy_status"] == "INSUFFICIENT_EVIDENCE"


def test_insufficient_cache_evidence_precedes_task_policy(prefix_case, tmp_path):
    prefix_case["mode"] = "missing_counts"
    recipe = decision_recipe(
        prefix_case,
        [{"id": "nonempty", "kind": "nonempty"}],
        [PolicyCriterion("quality.candidate_pass_rate", "1", minimum=0)],
    )
    result = run_recipe(recipe, tmp_path / "incomplete")
    assert result.exit_code == 4
    assert result.manifest["policy_status"] == "NOT_EVALUATED"


def test_reference_variation_reports_observed_differences_without_quality_claim(
    prefix_case, tmp_path
):
    from metria.recipes import _json_value

    recipe = decision_recipe(
        prefix_case,
        [],
        [PolicyCriterion("behavior.reference_repeatability", "1", minimum=0)],
    )
    result = run_recipe(recipe, tmp_path / "reference")
    record = load_run_record(result.output_dir / "reference.run.json")
    evidence = _json_value(record.evidence, path="evidence")
    evidence["measurements"][TokenTrajectoryProtocol.name]["prompts"][2][
        "token_ids"
    ] = [1, 2, 99]
    changed = replace(record, evidence=evidence)
    variability, metrics = reference_variability(changed)
    assert variability["different_pairs"] == 1
    assert metrics["reference_repeatability"] == 0.5
    assert "not a statistical guarantee" in variability["limitation"]


@pytest.mark.parametrize(
    "target,version,value",
    [
        ("performance.latency_ratio", "1", -0.1),
        ("performance.candidate_latency_seconds", "1", float("inf")),
        ("quality.candidate_pass_rate", "1", 1.1),
        ("quality.pass_rate_delta", "1", -1.1),
    ],
)
def test_each_target_has_its_own_valid_numeric_domain(target, version, value):
    with pytest.raises((ValueError, TypeError)):
        PolicyCriterion(target, version, maximum=value)


def test_latency_bounds_and_quality_deltas_are_not_forced_into_fraction_domain():
    assert (
        PolicyCriterion("performance.latency_ratio", "1", maximum=1.25).unit == "ratio"
    )
    assert (
        PolicyCriterion("performance.candidate_latency_seconds", "1", maximum=30).unit
        == "seconds"
    )
    assert (
        PolicyCriterion("quality.pass_rate_delta", "1", minimum=-0.05).unit
        == "fraction_delta"
    )


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "wrong_method",
        "coverage",
        "nonfinite",
        "wrong_count",
        "wrong_fingerprint",
        "wrong_passed",
    ],
)
def test_incomplete_or_incompatible_quality_evidence_is_not_a_score(
    prefix_case, tmp_path, damage
):
    from metria.measurements.verification_impact import compare_task_checks
    from metria.recipes import _json_value

    recipe = decision_recipe(
        prefix_case,
        [{"id": "answer", "kind": "nonempty"}],
        [PolicyCriterion("quality.candidate_pass_rate", "1", minimum=0)],
    )
    result = run_recipe(recipe, tmp_path / damage)
    left = load_run_record(result.output_dir / "reference.run.json")
    right = load_run_record(result.output_dir / "candidate.run.json")
    evidence = _json_value(right.evidence, path="evidence")
    quality = evidence["measurements"][TokenTrajectoryProtocol.name]["task_checks"]
    metrics = dict(right.metrics)
    if damage == "missing":
        metrics.pop("task_check_pass_rate")
    elif damage == "wrong_method":
        quality["method"] = "different"
    elif damage == "coverage":
        metrics["task_check_pass_rate"] = replace(
            metrics["task_check_pass_rate"], coverage=None
        )
    elif damage == "nonfinite":
        metrics["task_check_pass_rate"] = replace(
            metrics["task_check_pass_rate"], value=float("nan")
        )
    elif damage == "wrong_count":
        quality["check_count"] += 1
    elif damage == "wrong_fingerprint":
        quality["checks"][0]["definition_sha256"] = "unknown"
    else:
        quality["passed"] = 0
    changed = replace(right, evidence=evidence, metrics=metrics)
    summary, values = compare_task_checks(left, changed)
    assert summary["status"] == "unavailable" and values == {}


def test_single_reference_trial_cannot_claim_repeatability(prefix_case, tmp_path):
    from metria.recipes import _json_value

    recipe = decision_recipe(
        prefix_case,
        [],
        [PolicyCriterion("behavior.reference_repeatability", "1", minimum=0)],
    )
    result = run_recipe(recipe, tmp_path / "repeatability")
    record = load_run_record(result.output_dir / "reference.run.json")
    evidence = _json_value(record.evidence, path="evidence")
    evidence["measurements"][TokenTrajectoryProtocol.name]["workload"][
        "measured_trials"
    ] = 1
    summary, values = reference_variability(replace(record, evidence=evidence))
    assert summary["status"] == "unavailable" and values == {}
