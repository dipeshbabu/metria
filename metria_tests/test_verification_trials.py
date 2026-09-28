from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from metria import ComparisonPlan, RunSpec, StudyRecipe, StudySpec
from metria import verification_trials as trials
from metria.verification import VerificationResult


def _recipe():
    run = RunSpec(
        model={"id": "fixture"},
        runtime={"name": "fixture"},
        scenario={},
        measurements=("fixture",),
    )
    return StudyRecipe(
        study=StudySpec(
            name="trial-fixture", runs=(run, run), comparison=ComparisonPlan()
        ),
        measurement_configs={},
        environment={},
    )


@pytest.fixture
def fake_verifier(monkeypatch):
    calls = []
    data = {
        "scope": trials.VERIFICATION_SCOPE,
        "recipe_digest": "a" * 64,
        "verdict": "VERIFIED",
        "hardware": {"source": "fixture"},
        "records": {
            role: {
                "record_digest": role,
                "observed": {"model_sha256": "b" * 64, "provider_sha256": "c" * 64},
            }
            for role in ("reference", "candidate")
        },
        "analyses": [{"name": "fixture", "version": "1"}],
        "performance": {
            "available": True,
            "reference": 2.0,
            "candidate": 1.0,
            "metric": {"method": "fixture"},
            "methodology": {"workload": "fixed"},
        },
    }

    def verify(recipe, output):
        calls.append(output)
        output.mkdir()
        (output / "verification.json").write_text(json.dumps(data))
        return VerificationResult(
            output,
            deepcopy(data),
            {"VERIFIED": 0, "FAIL": 1, "NOT_COMPARABLE": 3, "EXECUTION_FAILED": 5}[
                data["verdict"]
            ],
        )

    monkeypatch.setattr(trials, "verify_recipe", verify)
    return calls, data


def test_warmups_are_retained_but_excluded_from_trial_samples(tmp_path, fake_verifier):
    calls, _ = fake_verifier
    output = tmp_path / "spaces and 'quotes' [data]"
    result = trials.execute_verification_trials(
        _recipe(), output, policy=trials.VerificationTrialPolicy(1, 2)
    )
    assert result["exit_code"] == 0 and len(calls) == 3
    assert [row["warmup"] for row in result["pairs"]] == [True, False, False]
    assert result["latency"]["sample_count"] == 2
    assert result["latency"]["reference"]["samples"] == [2.0, 2.0]
    assert (output / "warmup-0000/verification.json").exists()
    assert (output / "verification-trials.json").exists()


def test_changed_observed_hardware_invalidates_cached_baseline(tmp_path, fake_verifier):
    _, data = fake_verifier
    baseline = trials.execute_verification_trials(_recipe(), tmp_path / "baseline")
    data["hardware"] = {"source": "different fixture"}
    result = trials.execute_verification_trials(
        _recipe(), tmp_path / "candidate", baseline=baseline
    )
    assert result["exit_code"] == 3 and result["baseline_matches"] is False
    assert not result["latency"]["available"]


def test_failed_pair_stops_trials_without_zero_measurements(tmp_path, fake_verifier):
    calls, data = fake_verifier
    data["verdict"] = "EXECUTION_FAILED"
    result = trials.execute_verification_trials(
        _recipe(), tmp_path / "failed", policy=trials.VerificationTrialPolicy(0, 3)
    )
    assert result["exit_code"] == 5 and len(calls) == 1
    assert not result["latency"]["available"]
    assert (tmp_path / "failed/trial-0000/verification.json").exists()


def test_missing_native_timing_is_unavailable_not_zero(tmp_path, fake_verifier):
    _, data = fake_verifier
    data["performance"] = {"available": False}
    result = trials.execute_verification_trials(_recipe(), tmp_path / "untimed")
    assert result["status"] == "completed"
    assert not result["latency"]["available"]
    assert "reference" not in result["latency"]


def test_existing_results_are_never_overwritten(tmp_path, fake_verifier):
    marker = tmp_path / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        trials.execute_verification_trials(_recipe(), tmp_path)
    assert marker.read_text() == "keep"
    assert not fake_verifier[0]


@pytest.mark.parametrize("warmup,measured", [(True, 1), (-1, 1), (0, 0), (0, 1.5)])
def test_invalid_trial_counts_fail_before_execution(warmup, measured):
    with pytest.raises(ValueError):
        trials.VerificationTrialPolicy(warmup, measured)


def test_bash_entrypoints_are_argument_only_wrappers():
    root = Path(__file__).parents[1] / "tools/benchmarks"
    for name in ("turbo-quick-bench.sh", "turbo-realworld-bench.sh"):
        text = (root / name).read_text()
        assert '"$@"' in text and "exec " in text
        assert "python3 -c" not in text and "sleep " not in text
        assert "REF_PPL" not in text and "SERVER_PID" not in text


@pytest.mark.parametrize(
    "exception,code,status",
    [(KeyboardInterrupt, 130, "interrupted"), (OSError, 5, "execution_failed")],
)
def test_interrupted_or_failed_trial_retains_earlier_evidence(
    tmp_path, fake_verifier, monkeypatch, exception, code, status
):
    original = trials.verify_recipe
    calls = 0

    def fail_second(recipe, output):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise exception("private path or prompt")
        return original(recipe, output)

    monkeypatch.setattr(trials, "verify_recipe", fail_second)
    output = tmp_path / "partial"
    result = trials.execute_verification_trials(
        _recipe(), output, policy=trials.VerificationTrialPolicy(0, 3)
    )
    assert result["exit_code"] == code and result["status"] == status
    assert len(result["pairs"]) == 1
    assert (output / "trial-0000/verification.json").is_file()
    assert "private path" not in (output / "verification-trials.json").read_text()
    assert not result["latency"]["available"]


def test_observed_identity_drift_stops_later_pairs(
    tmp_path, fake_verifier, monkeypatch
):
    original = trials.verify_recipe
    calls, data = fake_verifier

    def drift(recipe, output):
        if calls:
            data["records"]["candidate"]["observed"]["model_sha256"] = "d" * 64
        return original(recipe, output)

    monkeypatch.setattr(trials, "verify_recipe", drift)
    result = trials.execute_verification_trials(
        _recipe(), tmp_path / "drift", policy=trials.VerificationTrialPolicy(0, 3)
    )
    assert result["status"] == "not_comparable" and result["exit_code"] == 3
    assert len(calls) == 2 and not result["latency"]["available"]


def test_matching_baseline_is_eligible_without_implying_a_speedup(
    tmp_path, fake_verifier
):
    baseline = trials.execute_verification_trials(_recipe(), tmp_path / "baseline")
    result = trials.execute_verification_trials(
        _recipe(), tmp_path / "repeat", baseline=baseline
    )
    assert result["baseline_matches"] is True and result["exit_code"] == 0
    assert "speedup" not in result


def test_policy_failure_keeps_samples_and_nonzero_exit(tmp_path, fake_verifier):
    _, data = fake_verifier
    data["verdict"] = "FAIL"
    result = trials.execute_verification_trials(
        _recipe(), tmp_path / "policy", policy=trials.VerificationTrialPolicy(1, 2)
    )
    assert result["status"] == "policy_failed" and result["exit_code"] == 1
    assert result["latency"]["sample_count"] == 2


@pytest.mark.parametrize(
    "changes", [{"fixture_only": True}, {"scope": "unknown"}, {"records": {}}]
)
def test_unqualified_identity_cannot_establish_baseline(
    tmp_path, fake_verifier, changes
):
    _, data = fake_verifier
    data.update(changes)
    result = trials.execute_verification_trials(_recipe(), tmp_path / "invalid")
    assert result["exit_code"] == 3 and result["baseline_key"] is None


@pytest.mark.parametrize("value", [True, -1, float("nan"), "1"])
def test_invalid_native_timing_is_not_aggregated(tmp_path, fake_verifier, value):
    _, data = fake_verifier
    data["performance"]["reference"] = value
    result = trials.execute_verification_trials(_recipe(), tmp_path / "invalid-time")
    assert not result["latency"]["available"]


def test_invalid_baseline_fails_before_output_creation(tmp_path, fake_verifier):
    output = tmp_path / "invalid-baseline"
    with pytest.raises(ValueError, match="schema"):
        trials.execute_verification_trials(_recipe(), output, baseline={})
    assert not output.exists() and not fake_verifier[0]
