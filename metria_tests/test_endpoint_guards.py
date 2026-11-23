"""Validation and malformed-response guards of the endpoint adapter."""

from __future__ import annotations

import pytest
from metria_tests.test_endpoint_runtime import _run, _spec, server  # noqa: F401

from metria import RunSpec
from metria.protocols import CaptureRequest, InferenceRequest
from metria.runtimes import endpoint
from metria.runtimes.endpoint import EndpointAdapter


def _with(spec: RunSpec, **changes) -> RunSpec:
    return RunSpec(**{**spec.__dict__, **changes})


@pytest.mark.parametrize(
    ("runtime", "match"),
    [
        ({"name": "vllm"}, "runtime.name"),
        ({"surprise": 1}, "unsupported endpoint runtime fields"),
        ({"flavor": "tgi"}, "runtime.flavor"),
        ({"base_url_env": "lower_case"}, "upper-case environment variable"),
        ({"api_key_env": "1BAD"}, "upper-case environment variable"),
        ({"timeout_s": 0}, "timeout_s"),
        ({"max_retries": 9}, "max_retries"),
    ],
)
def test_runtime_validation(tmp_path, runtime, match):
    spec = _spec(tmp_path)
    spec = _with(spec, runtime={**spec.runtime, **runtime})
    with pytest.raises(ValueError, match=match):
        endpoint._runtime(spec)


@pytest.mark.parametrize(
    ("model", "match"),
    [
        ({"id": ""}, "model.id"),
        ({"tokenizer_path": None}, "tokenizer_path"),
        ({"tokenizer_files": {"model.bin": "a" * 64}}, "known tokenizer files"),
    ],
)
def test_model_validation(tmp_path, model, match):
    spec = _spec(tmp_path)
    spec = _with(spec, model={**spec.model, **model})
    with pytest.raises(ValueError, match=match):
        endpoint._model(spec, "vllm")


def test_generic_flavor_needs_only_a_model_id(tmp_path):
    spec = _with(_spec(tmp_path), model={"id": "served"})
    assert endpoint._model(spec, "generic") == {"id": "served"}


@pytest.mark.parametrize(
    ("scenario", "flavor", "files", "match"),
    [
        ({"chat_template": "yes"}, "vllm", {}, "boolean"),
        ({"chat_template": True}, "generic", {}, "token-ID flavors"),
        (
            {"chat_template": True},
            "vllm",
            {"tokenizer.json": "a" * 64},
            "tokenizer_config",
        ),
    ],
)
def test_chat_template_request_validation(tmp_path, scenario, flavor, files, match):
    spec = _spec(tmp_path)
    spec = _with(
        spec,
        scenario={**spec.scenario, **scenario},
        model={**spec.model, "tokenizer_files": files},
    )
    with pytest.raises(ValueError, match=match):
        endpoint._chat_requested(spec, flavor)


@pytest.mark.parametrize("tokens", [0, 40000, True, "8"])
def test_max_tokens_is_bounded(tokens):
    with pytest.raises(ValueError, match="max_tokens"):
        endpoint._decode({"max_tokens": tokens})


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"choices": []}, "one choice"),
        ({"choices": [{"text": None}]}, "text is missing"),
        ([], "one choice"),
    ],
)
def test_malformed_completion_payloads(payload, match):
    with pytest.raises(RuntimeError, match=match):
        endpoint._choice(payload)


@pytest.mark.parametrize(
    "choice",
    [{"token_ids": None}, {"token_ids": [1, -2]}, {"token_ids": [True]}],
)
def test_malformed_token_ids(choice):
    with pytest.raises(RuntimeError, match="generated token IDs"):
        endpoint._token_ids(choice, [1])


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        ([["not a mapping"]], "malformed"),
        ([{"7": -0.1}], "keyed by token ID"),
        ([{"token_id:x": -0.1}], "malformed"),
        ([{"token_id:1": 0.5}], "finite and <= 0"),
        ([{"token_id:1": float("nan")}], "finite and <= 0"),
        ([{"token_id:1": -0.1}], "fewer than the requested"),
    ],
)
def test_malformed_top_logprobs(rows, match):
    choice = {
        "logprobs": {"top_logprobs": rows[0] if rows == [["not a mapping"]] else rows}
    }
    with pytest.raises(RuntimeError, match=match):
        endpoint._top_logprobs(choice, 1)


def test_session_guards(server, tmp_path):  # noqa: F811
    adapter = EndpointAdapter()
    session = adapter.launch(adapter.resolve(_spec(tmp_path), {}), {})
    with pytest.raises(RuntimeError, match="unsupported endpoint capture"):
        session.infer((InferenceRequest("a"),), capture=(CaptureRequest("hidden"),))
    with pytest.raises(RuntimeError, match="cannot reset"):
        session.reset("prefix-cache")
    session.reset("measurement")
    assert session.infer(()).metadata["invocations"] == ()
    with pytest.raises(TypeError, match="own sessions"):
        adapter.observe(object())


def test_generic_session_sends_text_and_refuses_token_captures(server, tmp_path):  # noqa: F811
    adapter = EndpointAdapter()
    spec = _with(_spec(tmp_path, flavor="generic"), model={"id": "served-model"})
    session = adapter.launch(adapter.resolve(spec, {}), {})
    with pytest.raises(RuntimeError, match="cannot return token captures"):
        session.infer((InferenceRequest("a"),), capture=(CaptureRequest("token_ids"),))
    with pytest.raises(RuntimeError, match="token-ID flavor"):
        session.infer_concurrent((InferenceRequest("a"),), max_in_flight=2)


def test_observe_records_unstable_identity_when_server_vanishes(server, tmp_path):  # noqa: F811
    adapter = EndpointAdapter()
    session = adapter.launch(adapter.resolve(_spec(tmp_path), {}), {})
    server.close()
    observed = adapter.observe(session)
    assert observed["identity_snapshots"]["after"] is None
    assert observed["identity_snapshots"]["stable"] is False
