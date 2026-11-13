"""OpenAI-compatible endpoint adapter against a real loopback HTTP server."""

from __future__ import annotations

import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from metria import RunSpec, execute_run
from metria.integrations.staged_captures import STAGED_CAPTURE_CHECK
from metria.measurements import TokenDistributionProtocol, TokenTrajectoryProtocol
from metria.records import run_record_to_json
from metria.runtimes import endpoint, endpoint_http
from metria.runtimes.endpoint import EndpointAdapter
from metria.runtimes.endpoint_http import parse_base_url

KEY = "sk-test-secret-value-0123456789"
PROMPTS = [{"id": "a", "prompt": "ab"}, {"id": "b", "prompt": "abc"}]


def _tokenize(text: str) -> list[int]:
    return [ord(char) for char in text]


class Server:
    """Scriptable vLLM-flavored completions server."""

    def __init__(self) -> None:
        self.mode = "ok"
        self.failures = 0
        self.sleep = 0.0
        self.headers: list[str | None] = []
        self.version = "0.30.0"
        self.root = "org/model"
        self.tokenize_status = 200
        self.tokenize_offset = 0
        self.completions = 0
        self.drift_after: int | None = None
        self.bodies: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, payload):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                outer.headers.append(self.headers.get("Authorization"))
                if self.path == "/v1/models":
                    self._send(
                        200,
                        {
                            "data": [
                                {
                                    "id": "served-model",
                                    "root": outer.root,
                                    "max_model_len": 256,
                                }
                            ]
                        },
                    )
                elif self.path == "/version":
                    self._send(200, {"version": outer.version})
                else:
                    self._send(404, {"error": "missing"})

            def do_POST(self):
                outer.headers.append(self.headers.get("Authorization"))
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/tokenize":
                    if outer.tokenize_status != 200:
                        self._send(outer.tokenize_status, {"error": "off"})
                        return
                    tokens = [ord(c) + outer.tokenize_offset for c in body["prompt"]]
                    self._send(200, {"tokens": tokens, "count": len(tokens)})
                    return
                outer.bodies.append(body)
                outer.completions += 1
                if (
                    outer.drift_after is not None
                    and outer.completions >= outer.drift_after
                ):
                    outer.version = "0.31.0"
                if outer.failures:
                    outer.failures -= 1
                    self._send(503, {"error": "busy"})
                    return
                time.sleep(outer.sleep)
                self._send(200, outer.completion(body))

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def completion(self, body: dict) -> dict:
        prompt = body["prompt"]
        generated = [prompt[-1] + 1, prompt[-1] + 2, 7]
        choice: dict = {
            "text": "out",
            "finish_reason": "length",
            "token_ids": generated,
        }
        choice["prompt_token_ids"] = prompt if self.mode != "prompt_mismatch" else [1]
        if body.get("logprobs"):
            width = body["logprobs"] + 1  # vLLM may add the sampled token
            rows = [
                {f"token_id:{100 + rank}": -0.1 * (rank + 1) for rank in range(width)}
                for _ in generated
            ]
            if self.mode == "short_logprobs":
                rows = rows[:-1]
            choice["logprobs"] = {
                "top_logprobs": rows,
                "tokens": [],
                "token_logprobs": [],
            }
        return {"choices": [choice]}

    def close(self) -> None:
        self.httpd.shutdown()


@pytest.fixture
def server(monkeypatch, tmp_path):
    instance = Server()
    monkeypatch.setenv("METRIA_TEST_URL", instance.url)
    monkeypatch.setenv("METRIA_TEST_KEY", KEY)
    monkeypatch.setattr(endpoint, "load_tokenizer", lambda path, files: _tokenize)
    monkeypatch.setattr(endpoint_http, "_backoff", lambda seconds: None)
    yield instance
    instance.close()


def _spec(
    tmp_path: Path, *, flavor="vllm", measurement=TokenTrajectoryProtocol, **runtime
):
    return RunSpec(
        model={
            "id": "served-model",
            "tokenizer_path": str(tmp_path),
            "tokenizer_files": {"tokenizer.json": "a" * 64},
        },
        runtime={
            "name": "openai_endpoint",
            "flavor": flavor,
            "base_url_env": "METRIA_TEST_URL",
            "api_key_env": "METRIA_TEST_KEY",
            "timeout_s": 5,
            "max_retries": 2,
            **runtime,
        },
        scenario={"max_tokens": 3, "temperature": 0.0, "seed": 1},
        measurements=(measurement.name,),
        trial_policy=(
            {"capability_overrides": [STAGED_CAPTURE_CHECK]}
            if measurement is TokenDistributionProtocol
            else {}
        ),
    )


def _run(spec, measurement=None):
    measurement = measurement or TokenTrajectoryProtocol()
    return execute_run(
        study_name="endpoint",
        run_id="run",
        spec=spec,
        adapter=EndpointAdapter(),
        measurement=measurement,
        measurement_config={"prompts": PROMPTS},
        environment={},
    )


def test_token_ids_are_captured_and_the_key_is_never_retained(server, tmp_path):
    record = _run(_spec(tmp_path))
    assert record.status.value == "completed", record.events
    rows = record.evidence["measurements"][TokenTrajectoryProtocol.name]["prompts"]
    assert [list(row["token_ids"]) for row in rows] == [[99, 100, 7], [100, 101, 7]]
    body = server.bodies[0]
    assert body["prompt"] == [97, 98] and body["return_token_ids"] is True
    assert "logprobs" not in body and body["seed"] == 1 and body["temperature"] == 0.0
    assert set(server.headers) == {f"Bearer {KEY}"}
    text = run_record_to_json(record)
    assert KEY not in text and server.url not in text
    assert record.observed["endpoint"]["host"] == "127.0.0.1"
    assert all(row["seconds"] >= 0 for row in record.observed["invocations"])


def test_top_logprobs_keep_exactly_k_highest(server, tmp_path):
    record = _run(
        _spec(tmp_path, measurement=TokenDistributionProtocol),
        TokenDistributionProtocol(),
    )
    assert record.status.value == "completed", record.events
    prompt = record.evidence["measurements"][TokenDistributionProtocol.name]["prompts"][
        0
    ]
    first = prompt["top_logprobs"][0]
    assert len(first) == 20 and [pair[0] for pair in first] == list(range(100, 120))
    assert math.isclose(first[0][1], -0.1)
    assert server.bodies[0]["return_tokens_as_token_ids"] is True
    assert (
        record.observed["invocations"][0]["top_logprobs"]["mode"] == "server_reported"
    )


def test_distributions_require_the_staged_override(server, tmp_path):
    spec = _spec(tmp_path, measurement=TokenDistributionProtocol)
    record = _run(
        RunSpec(**{**spec.__dict__, "trial_policy": {}}), TokenDistributionProtocol()
    )
    assert record.status.value == "preflight_failed"


def test_transient_errors_are_retried_and_retained(server, tmp_path):
    server.failures = 1
    record = _run(_spec(tmp_path))
    assert record.status.value == "completed", record.events
    attempts = record.observed["invocations"][0]["attempts"]
    assert [row["status"] for row in attempts] == [503, 200]


def test_exhausted_retries_fail_the_run(server, tmp_path):
    server.failures = 10
    record = _run(_spec(tmp_path, max_retries=1))
    assert record.status.value == "failed"
    assert KEY not in run_record_to_json(record)


@pytest.mark.parametrize("mode", ["short_logprobs", "prompt_mismatch"])
def test_incomplete_or_inconsistent_responses_fail(server, tmp_path, mode):
    server.mode = mode
    measurement = (
        TokenDistributionProtocol
        if mode == "short_logprobs"
        else TokenTrajectoryProtocol
    )
    record = _run(_spec(tmp_path, measurement=measurement), measurement())
    assert record.status.value == "failed"


def test_timeout_is_a_transport_failure(server, tmp_path):
    server.sleep = 1.5
    record = _run(_spec(tmp_path, timeout_s=0.5, max_retries=0))
    assert record.status.value == "failed"


def test_unserved_model_fails_at_launch(server, tmp_path):
    spec = _spec(tmp_path)
    record = _run(RunSpec(**{**spec.__dict__, "model": {**spec.model, "id": "other"}}))
    assert record.status.value == "failed"
    assert any(event.get("stage") == "launch" for event in record.events)


def test_missing_environment_variable_is_unsupported(server, tmp_path, monkeypatch):
    monkeypatch.delenv("METRIA_TEST_KEY")
    assert _run(_spec(tmp_path)).status.value == "preflight_failed"


def test_generic_flavor_cannot_supply_token_captures(server, tmp_path):
    assert _run(_spec(tmp_path, flavor="generic")).status.value == "preflight_failed"


@pytest.mark.parametrize(
    ("url", "match"),
    [
        ("http://inference.internal:8000", "loopback"),
        ("https://user:pw@example.com", "credentials"),
        ("https://example.com/v1?key=x", "query"),
        ("ftp://example.com", "http"),
    ],
)
def test_unsafe_urls_are_rejected(url, match):
    with pytest.raises(ValueError, match=match):
        parse_base_url(url)


def test_https_and_loopback_urls_are_accepted():
    assert parse_base_url("https://example.com/proxy").hostname == "example.com"
    assert parse_base_url("http://[::1]:8000").hostname == "::1"


def test_tokenizer_pins_are_validated_and_hashed(tmp_path):
    import hashlib

    content = b'{"model": {}}'
    (tmp_path / "tokenizer.json").write_bytes(content)
    good = {"tokenizer.json": hashlib.sha256(content).hexdigest()}
    with pytest.raises(ValueError, match="known tokenizer files"):
        endpoint.tokenizer_pins({"model.safetensors": "a" * 64})
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        endpoint.tokenizer_pins({"tokenizer.json": "XYZ"})
    with pytest.raises(RuntimeError, match="does not match"):
        endpoint.load_tokenizer(str(tmp_path), {"tokenizer.json": "0" * 64})
    try:
        import tokenizers  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="tokenizers"):
            endpoint.load_tokenizer(str(tmp_path), good)
