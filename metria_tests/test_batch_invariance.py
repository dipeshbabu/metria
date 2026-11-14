"""Batch invariance under a declared concurrent schedule on endpoint servers."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from metria_tests.test_endpoint_verify import PROMPTS, _verify, servers  # noqa: F401

from metria.measurements import batch_invariance as invariance
from metria.measurements.prefix_workload import trajectory_config
from metria.policies import PolicyCriterion, VerificationPolicy
from metria.preparation_endpoint import prepare_endpoint_recipe
from metria.records import load_run_record
from metria.verification_endpoint import validate_endpoint_recipe

POLICY = VerificationPolicy(
    (
        PolicyCriterion("quality.candidate_pass_rate", "1", minimum=1.0),
        PolicyCriterion("behavior.candidate_batch_invariance", "1", minimum=1.0),
        PolicyCriterion("client.request_latency_ratio", "1", maximum=1000.0),
    )
)


def _recipe(tmp_path, *, max_in_flight=4, rounds=2, policy=POLICY):
    return prepare_endpoint_recipe(
        model_id="served-model",
        tokenizer_dir=tmp_path / "tok",
        prompts=PROMPTS,
        reference_url_env="REF_URL",
        candidate_url_env="CAND_URL",
        reference_declared={"kv_cache_dtype": "auto"},
        candidate_declared={"kv_cache_dtype": "fp8"},
        policy=policy,
        max_tokens=3,
        concurrency=max_in_flight,
        concurrency_rounds=rounds,
    )


def test_invariant_servers_pass_under_concurrency(servers, tmp_path):  # noqa: F811
    for server in servers:
        server.sleep = 0.05
    result = _verify(_recipe(tmp_path), tmp_path / "out")
    manifest = result.manifest
    assert manifest["verdict"] == "PASS", json.dumps(manifest, default=str)[:3000]
    assert all(server.max_seen_in_flight > 1 for server in servers)
    rows = {r["criterion"]["target"]: r for r in manifest["policy"]["criteria"]}
    assert rows["behavior.candidate_batch_invariance"]["observed_value"] == 1.0
    candidate = load_run_record(result.output_dir / "candidate.run.json")
    capture = candidate.evidence["measurements"]["kv_fidelity.decode_time_trajectory"]
    evidence = capture["batch_invariance"]
    assert evidence["requests"] == 4 and evidence["matched"] == 4
    assert evidence["client"]["boundary"].endswith("includes the network")
    assert len(candidate.observed["concurrent_invocations"]) == 4
    assert candidate.requested.trial_policy["concurrency"] == {
        "max_in_flight": 4,
        "rounds": 2,
    }


def test_batch_variant_candidate_fails(servers, tmp_path):  # noqa: F811
    for server in servers:
        server.sleep = 0.05
    servers[1].batch_variant = True
    manifest = _verify(_recipe(tmp_path), tmp_path / "out").manifest
    assert manifest["verdict"] == "FAIL"
    rows = {r["criterion"]["target"]: r for r in manifest["policy"]["criteria"]}
    assert rows["behavior.candidate_batch_invariance"]["observed_value"] < 1.0
    analysis = next(
        a for a in manifest["analyses"] if a["name"] == "metria.batch_invariance"
    )
    assert analysis["metrics"]["reference_batch_invariance"]["value"] == 1.0


def test_concurrency_requires_its_analysis(servers, tmp_path):  # noqa: F811
    recipe = _recipe(
        tmp_path,
        policy=VerificationPolicy(
            (PolicyCriterion("quality.candidate_pass_rate", "1", minimum=1.0),)
        ),
    )
    plan = replace(
        recipe.study.comparison, analyses=recipe.study.comparison.analyses[:2]
    )
    with pytest.raises(ValueError, match="batch_invariance exactly when"):
        validate_endpoint_recipe(
            replace(recipe, study=replace(recipe.study, comparison=plan))
        )


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ({"max_in_flight": 1}, "from 2 to 64"),
        ({"max_in_flight": 65}, "from 2 to 64"),
        ({"max_in_flight": 4, "rounds": 0}, "from 1 to 10"),
        ({"max_in_flight": 4, "burst": 2}, "accepts only"),
        ({"rounds": 2}, "max_in_flight"),
    ],
)
def test_concurrency_settings_are_bounded(value, match):
    with pytest.raises(ValueError, match=match):
        invariance.concurrency_settings({"concurrency": value})


def test_local_profiles_reject_concurrency():
    with pytest.raises(ValueError, match="unsupported prefix-workload"):
        trajectory_config(
            {
                "prompts": [{"id": "a", "prompt": "x"}],
                "concurrency": {"max_in_flight": 2},
            }
        )


def test_invariance_targets_are_bounded():
    with pytest.raises(ValueError, match="target domain"):
        PolicyCriterion("behavior.candidate_batch_invariance", "1", minimum=1.5)
    assert PolicyCriterion("client.requests_per_second_ratio", "1", minimum=0.8)
