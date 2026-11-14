"""Seeded sampling, stop sequences, chat templates and pass-rate intervals."""

from __future__ import annotations

import json

import pytest
from metria_tests.test_endpoint_verify import PROMPTS, _verify, servers  # noqa: F401

from metria.measurements.endpoint_sampling import expand_seeds, validate_scenario
from metria.measurements.pass_rate_intervals import newcombe, wilson
from metria.policies import PolicyCriterion, VerificationPolicy
from metria.preparation_endpoint import prepare_endpoint_recipe
from metria.records import load_run_record
from metria.runtimes import endpoint_chat

POLICY = VerificationPolicy(
    (
        PolicyCriterion("quality.candidate_pass_rate_lower_bound", "1", minimum=0.5),
        PolicyCriterion("quality.pass_rate_delta_lower_bound", "1", minimum=-0.6),
    )
)


def _recipe(tmp_path, **options):
    return prepare_endpoint_recipe(
        model_id="served-model",
        tokenizer_dir=tmp_path / "tok",
        prompts=PROMPTS,
        reference_url_env="REF_URL",
        candidate_url_env="CAND_URL",
        reference_declared={"kv_cache_dtype": "auto"},
        candidate_declared={"kv_cache_dtype": "fp8"},
        policy=POLICY,
        max_tokens=3,
        **options,
    )


def test_seeded_sampling_expands_prompts_and_reports_intervals(servers, tmp_path):  # noqa: F811
    recipe = _recipe(
        tmp_path, temperature=0.7, seeds=[3, 5, 8], top_p=0.9, stop=["\n\n"]
    )
    result = _verify(recipe, tmp_path / "out")
    manifest = result.manifest
    assert manifest["verdict"] == "PASS", json.dumps(manifest, default=str)[:2500]
    bodies = servers[1].bodies
    assert sorted({body["seed"] for body in bodies}) == [3, 5, 8]
    assert all(body["temperature"] == 0.7 and body["top_p"] == 0.9 for body in bodies)
    assert all(body["stop"] == ["\n\n"] for body in bodies)
    candidate = load_run_record(result.output_dir / "candidate.run.json")
    capture = candidate.evidence["measurements"]["kv_fidelity.decode_time_trajectory"]
    ids = {row["id"].partition(":")[2] for row in capture["prompts"]}
    assert ids == {f"{p}@seed{s}" for p in ("a", "b") for s in (3, 5, 8)}
    impact = next(
        a for a in manifest["analyses"] if a["name"] == "metria.verification_impact"
    )
    lower = impact["metrics"]["candidate_task_pass_rate_lower"]["value"]
    assert 0.5 < lower < 1.0
    assert (
        impact["metrics"]["task_pass_rate_delta_lower"]["value"]
        < 0
        < (impact["metrics"]["task_pass_rate_delta_upper"]["value"])
    )


class FakeChat:
    def __init__(self, path, files):
        self.sha256 = "c" * 64

    def render(self, prompt):
        return f"<user>{prompt}</user>"

    def evidence(self):
        return {"status": "verified", "source": "test", "sha256": self.sha256}


def test_chat_template_is_rendered_locally(servers, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(endpoint_chat, "ChatTemplate", FakeChat)
    (tmp_path / "tok" / "tokenizer_config.json").write_text("{}")
    result = _verify(_recipe(tmp_path, chat_template=True), tmp_path / "out")
    assert result.manifest["verdict"] == "PASS", result.manifest["records"]
    sent = servers[0].bodies[0]["prompt"]
    assert sent == [ord(c) for c in "<user>ab</user>"]
    candidate = load_run_record(result.output_dir / "candidate.run.json")
    chat = candidate.observed["identity"]["chat_template"]
    assert chat["status"] == "verified" and chat["sha256"] == "c" * 64


def test_chat_template_requires_a_pinned_config(servers, tmp_path):  # noqa: F811
    with pytest.raises(ValueError, match="tokenizer_config.json"):
        _verify(_recipe(tmp_path, chat_template=True), tmp_path / "out")


def test_real_chat_template_rendering(tmp_path):
    pytest.importorskip("jinja2")
    import hashlib

    config = {
        "chat_template": "{% for m in messages %}<|im_start|>{{ m.role }}\n"
        "{{ m.content }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
        "bos_token": "<s>",
    }
    raw = json.dumps(config).encode()
    (tmp_path / "tokenizer_config.json").write_bytes(raw)
    files = {"tokenizer_config.json": hashlib.sha256(raw).hexdigest()}
    chat = endpoint_chat.ChatTemplate(str(tmp_path), files)
    assert chat.render("hi") == (
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    )


@pytest.mark.parametrize(
    ("scenario", "match"),
    [
        ({"max_tokens": 3, "temperature": 0.7, "seed": 1}, "1-32 distinct"),
        ({"max_tokens": 3, "temperature": 0.7, "seeds": [1, 1]}, "1-32 distinct"),
        ({"max_tokens": 3, "temperature": 0, "seeds": [1]}, "one integer seed"),
        ({"max_tokens": 3, "temperature": 3, "seed": 1}, "from 0 to 2"),
        ({"max_tokens": 3, "temperature": 0, "seed": 1, "top_p": 0}, "top_p"),
        (
            {"max_tokens": 3, "temperature": 0, "seed": 1, "stop": ["a"] * 5},
            "at most 4",
        ),
        ({"max_tokens": 3, "temperature": 0, "seed": 1, "beam": 2}, "unsupported"),
    ],
)
def test_scenarios_are_validated(scenario, match):
    with pytest.raises(ValueError, match=match):
        validate_scenario(scenario)


def test_expand_seeds_keeps_checks_and_sets_per_request_seed():
    config = {"prompts": [{"id": "a", "prompt": "x", "checks": [{"id": "c"}]}]}
    expanded = expand_seeds(config, {"seeds": [1, 2]})
    assert [row["id"] for row in expanded["prompts"]] == ["a@seed1", "a@seed2"]
    assert expanded["prompts"][1]["generation"] == {"seed": 2}
    assert expanded["prompts"][0]["checks"] == [{"id": "c"}]
    assert expand_seeds(config, {"seed": 1}) == config


def test_intervals_match_published_examples():
    low, high = wilson(81, 263)
    assert round(low, 4) == 0.2553 and round(high, 4) == 0.3662
    low, high = newcombe(56, 70, 48, 80)
    assert round(low, 4) == -0.3339 and round(high, 4) == -0.0524
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == 1.0
