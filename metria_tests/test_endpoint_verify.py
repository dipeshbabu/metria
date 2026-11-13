"""remote_endpoint_change.v1 end to end against two loopback servers."""

from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path

import pytest
from metria_tests.test_endpoint_runtime import KEY, Server, _tokenize

from metria import verification
from metria.cli import main
from metria.policies import PolicyCriterion, VerificationPolicy
from metria.preparation_endpoint import prepare_endpoint_recipe
from metria.recipes import StudyRecipe
from metria.runtimes import endpoint, endpoint_http
from metria.verification_endpoint import build_route, validate_endpoint_recipe

PROMPTS = [
    {"id": "a", "prompt": "ab", "checks": [{"id": "answered", "kind": "nonempty"}]},
    {"id": "b", "prompt": "abc", "checks": [{"id": "answered", "kind": "nonempty"}]},
]
POLICY = VerificationPolicy(
    (
        PolicyCriterion("quality.candidate_pass_rate", "1", minimum=1.0),
        PolicyCriterion("behavior.trajectory_agreement", "0.3.4", minimum=1.0),
    )
)


@pytest.fixture
def servers(monkeypatch, tmp_path):
    reference, candidate = Server(), Server()
    monkeypatch.setenv("REF_URL", reference.url)
    monkeypatch.setenv("CAND_URL", candidate.url)
    monkeypatch.setenv("REF_KEY", KEY)
    monkeypatch.setenv("CAND_KEY", KEY)
    monkeypatch.setattr(endpoint, "load_tokenizer", lambda path, files: _tokenize)
    monkeypatch.setattr(endpoint_http, "_backoff", lambda seconds: None)
    (tmp_path / "tok").mkdir()
    (tmp_path / "tok" / "tokenizer.json").write_text('{"model": {}}')
    yield reference, candidate
    reference.close()
    candidate.close()


def _recipe(tmp_path: Path, *, policy=POLICY, kld=False, **declared) -> StudyRecipe:
    reference = {"kv_cache_dtype": "auto", **declared.pop("reference", {})}
    candidate = {"kv_cache_dtype": "fp8", **declared.pop("candidate", {})}
    return prepare_endpoint_recipe(
        model_id="served-model",
        tokenizer_dir=tmp_path / "tok",
        prompts=PROMPTS,
        reference_url_env="REF_URL",
        candidate_url_env="CAND_URL",
        reference_key_env="REF_KEY",
        candidate_key_env="CAND_KEY",
        reference_declared=reference,
        candidate_declared=candidate,
        policy=policy,
        max_tokens=3,
        staged_distributions=kld,
    )


def _verify(recipe: StudyRecipe, output: Path):
    return verification._verify_with_profile(recipe, output, build_route(recipe))


def test_matching_servers_pass_with_claimed_identity(servers, tmp_path):
    result = _verify(_recipe(tmp_path), tmp_path / "out")
    manifest = result.manifest
    assert manifest["verdict"] == "PASS", json.dumps(manifest, default=str)[:3000]
    assert manifest["scope"] == "remote_endpoint_change.v1"
    assert manifest["change"]["identity_authority"] == "claimed"
    assert manifest["change"]["candidate"] == {"kv_cache_dtype": "fp8"}
    facts = manifest["records"]["candidate"]["observed"]
    assert facts["identity_authority"] == "claimed"
    assert facts["claimed_runtime_version"] == "0.30.0"
    report = (result.output_dir / "report.md").read_text()
    assert "not verified by Metria" in report
    assert "Declared endpoint configuration" in report
    for path in result.output_dir.iterdir():
        text = path.read_text()
        assert KEY not in text and servers[0].url not in text, path.name


def test_changed_trajectories_fail_the_policy(servers, tmp_path):
    servers[1].token_offset = 5
    assert _verify(_recipe(tmp_path), tmp_path / "out").manifest["verdict"] == "FAIL"


def test_undeclared_server_difference_is_not_comparable(servers, tmp_path):
    servers[1].version = "0.31.0"
    manifest = _verify(_recipe(tmp_path), tmp_path / "out").manifest
    assert manifest["verdict"] == "NOT_COMPARABLE"
    assert any(
        "runtime.version" in issue["dimension"]
        for issue in manifest["comparison"]["issues"]
    )


def test_declared_engine_upgrade_is_comparable(servers, tmp_path):
    servers[1].version = "0.31.0"
    recipe = _recipe(
        tmp_path,
        reference={"engine_version": "0.30.0"},
        candidate={"engine_version": "0.31.0"},
    )
    assert _verify(recipe, tmp_path / "out").manifest["verdict"] == "PASS"


def test_identity_drift_is_insufficient_evidence(servers, tmp_path):
    servers[1].drift_after = 1
    manifest = _verify(_recipe(tmp_path), tmp_path / "out").manifest
    assert manifest["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_declared_contradiction_fails_execution(servers, tmp_path):
    recipe = _recipe(
        tmp_path,
        reference={"engine_version": "0.30.0"},
        candidate={"engine_version": "0.29.0"},
    )
    manifest = _verify(recipe, tmp_path / "out").manifest
    assert manifest["verdict"] == "EXECUTION_FAILED"


def test_staged_kld_policy_runs_on_endpoints(servers, tmp_path):
    policy = VerificationPolicy(
        (
            PolicyCriterion("quality.candidate_pass_rate", "1", minimum=1.0),
            PolicyCriterion("behavior.trajectory_kld_mean", "0.1.0", maximum=1e-9),
        )
    )
    manifest = _verify(
        _recipe(tmp_path, policy=policy, kld=True), tmp_path / "o"
    ).manifest
    assert manifest["verdict"] == "PASS", json.dumps(manifest, default=str)[:2000]


def test_invalid_endpoint_recipes_are_rejected(servers, tmp_path):
    recipe = _recipe(tmp_path)
    with pytest.raises(ValueError, match="acceptance policy"):
        validate_endpoint_recipe(replace(recipe, policy=None))
    with pytest.raises(ValueError, match="must be empty"):
        validate_endpoint_recipe(replace(recipe, environment={"x": 1}))
    plan = replace(recipe.study.comparison, vary=frozenset({"runtime.base_url_env"}))
    with pytest.raises(ValueError, match="comparison.vary"):
        validate_endpoint_recipe(
            replace(recipe, study=replace(recipe.study, comparison=plan))
        )
    with pytest.raises(ValueError, match="different URL variables"):
        prepare_endpoint_recipe(
            model_id="served-model",
            tokenizer_dir=tmp_path / "tok",
            prompts=PROMPTS,
            reference_url_env="REF_URL",
            candidate_url_env="REF_URL",
            reference_declared={},
            candidate_declared={},
            policy=POLICY,
        )
    with pytest.raises(ValueError, match="task checks"):
        prepare_endpoint_recipe(
            model_id="served-model",
            tokenizer_dir=tmp_path / "tok",
            prompts=[{"id": "a", "prompt": "ab"}],
            reference_url_env="REF_URL",
            candidate_url_env="CAND_URL",
            reference_declared={},
            candidate_declared={},
            policy=POLICY,
        )


def test_prepare_endpoint_cli_writes_no_secrets(servers, tmp_path):
    workload = tmp_path / "w.jsonl"
    workload.write_text("\n".join(json.dumps(row) for row in PROMPTS))
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps(POLICY.to_data()))
    output = tmp_path / "recipe.json"
    out = io.StringIO()
    code = main(
        [
            "recipe", "prepare-endpoint",
            "--model-id", "served-model",
            "--reference-url-env", "REF_URL",
            "--candidate-url-env", "CAND_URL",
            "--tokenizer", str(tmp_path / "tok"),
            "--workload", str(workload),
            "--policy", str(policy),
            "--reference-declared", "kv_cache_dtype=auto",
            "--candidate-declared", "kv_cache_dtype=fp8",
            "--output", str(output),
        ],
        stdout=out,
    )  # fmt: skip
    assert code == 0, out.getvalue()
    text = output.read_text()
    assert servers[0].url not in text and KEY not in text
    assert '"runtime.declared.kv_cache_dtype"' in text
    verified = io.StringIO()
    exit_code = main(
        ["verify", str(output), "--output", str(tmp_path / "verification")],
        stdout=verified,
    )
    assert exit_code == 0, verified.getvalue()
    manifest = json.loads((tmp_path / "verification" / "verification.json").read_text())
    assert manifest["scope"] == "remote_endpoint_change.v1"
    assert manifest["verdict"] == "PASS"
