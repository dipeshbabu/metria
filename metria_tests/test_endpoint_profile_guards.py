"""Rejection paths and evidence gaps of remote_endpoint_change.v1."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from metria_tests.test_endpoint_verify import _recipe, servers  # noqa: F401

from metria.measurements import TokenTrajectoryProtocol
from metria.models import TreatmentSpec, TreatmentType
from metria.verification_endpoint import (
    ISOLATION,
    _concurrency_gaps,
    _distribution_gaps,
    _workload_gaps,
    derived_vary,
    validate_endpoint_recipe,
)

TRAJECTORY = TokenTrajectoryProtocol.name


def _runs(recipe, mutate):
    left, right = recipe.study.runs
    return replace(recipe, study=replace(recipe.study, runs=mutate(left, right)))


def _runtime(run, **changes):
    return replace(run, runtime={**run.runtime, **changes})


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda ref, cand: (ref,), "exactly two runs"),
        (lambda ref, cand: (ref, _runtime(cand, name="vllm")), "two endpoint runs"),
        (
            lambda ref, cand: (ref, _runtime(cand, flavor="generic")),
            "returns token IDs",
        ),
        (
            lambda ref, cand: (
                ref,
                replace(
                    cand,
                    treatments=(
                        TreatmentSpec(
                            "vllm.kv_cache",
                            TreatmentType.RUNTIME_FEATURE,
                            {"dtype": "fp8"},
                        ),
                    ),
                ),
            ),
            "treatments or placement",
        ),
        (lambda ref, cand: (ref, replace(cand, trial_policy={})), "trial policy"),
        (
            lambda ref, cand: (ref, replace(cand, measurements=("other",))),
            "exactly the trajectory",
        ),
        (lambda ref, cand: (ref, _runtime(cand, timeout_s=5)), "differ outside"),
        (
            lambda ref, cand: (
                ref,
                _runtime(cand, base_url_env=ref.runtime["base_url_env"]),
            ),
            "different URL",
        ),
    ],
)
def test_run_shape_rejections(servers, tmp_path, mutate, match):  # noqa: F811
    recipe = _runs(_recipe(tmp_path), mutate)
    with pytest.raises(ValueError, match=match):
        validate_endpoint_recipe(recipe)


def test_extra_measurement_config_is_rejected(servers, tmp_path):  # noqa: F811
    recipe = _recipe(tmp_path)
    configs = {**recipe.measurement_configs, "other": {}}
    with pytest.raises(ValueError):
        validate_endpoint_recipe(replace(recipe, measurement_configs=configs))


def test_declared_keys_must_be_stated_on_both_runs(servers, tmp_path):  # noqa: F811
    left, right = _recipe(tmp_path).study.runs
    right = _runtime(right, declared={"kv_cache_dtype": "fp8", "hardware": "L4"})
    with pytest.raises(ValueError, match="stated for both runs"):
        derived_vary(left, right)
    same = _runtime(right, declared=dict(left.runtime["declared"]))
    assert "runtime.declared.kv_cache_dtype" not in derived_vary(left, same)


def _record(capture=None, invocations=(), concurrent=()):
    return SimpleNamespace(
        evidence={"measurements": {TRAJECTORY: capture or {}}},
        observed={
            "invocations": list(invocations),
            "concurrent_invocations": list(concurrent),
        },
    )


def _workload(samples, measured=1, prompts=1):
    return {
        "n_prompts": prompts,
        "workload": {
            "method": "metria.prefix_cache_trials.v1",
            "isolation": ISOLATION,
            "warmup_trials": 0,
            "measured_trials": measured,
            "samples": samples,
        },
    }


TRIALS = {"warmup_trials": 0, "measured_trials": 1}


def test_workload_gaps():
    assert _workload_gaps(_record(), TRIALS) == [
        "declared endpoint trial evidence is missing or inconsistent"
    ]
    sample = {"prompt_tokens": 3}
    mismatch = _record(_workload([sample]), invocations=[])
    assert "sample counts" in _workload_gaps(mismatch, TRIALS)[0]
    wrong = _record(
        _workload([sample]), invocations=[{"completion_tokens": 2, "prompt_tokens": 4}]
    )
    assert "request evidence" in _workload_gaps(wrong, TRIALS)[0]
    good = _record(
        _workload([sample]), invocations=[{"completion_tokens": 2, "prompt_tokens": 3}]
    )
    assert _workload_gaps(good, TRIALS) == []


def test_distribution_and_concurrency_gaps():
    assert _distribution_gaps(_record({"n_prompts": 2})) == [
        "staged distribution evidence is missing or incomplete"
    ]
    assert (
        _distribution_gaps(_record({"n_prompts": 2, "distributions": {"n_prompts": 2}}))
        == []
    )
    settings = {"max_in_flight": 4, "rounds": 2}
    capture = {**_workload([], measured=2, prompts=4)}
    assert _concurrency_gaps(_record(capture), settings)
    capture["batch_invariance"] = {"max_in_flight": 4, "requests": 4}
    assert _concurrency_gaps(_record(capture, concurrent=[{}] * 3), settings)
    assert _concurrency_gaps(_record(capture, concurrent=[{}] * 4), settings) == []
