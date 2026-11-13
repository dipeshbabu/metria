"""OpenAI-compatible HTTP endpoint adapter for already-running inference servers.

Metria does not launch the server. Prompts are tokenized locally with a
hash-pinned tokenizer and sent as token IDs, so the server's own tokenization
and chat template cannot silently change the request. Token IDs and top-k
logprobs come from the response; latency is measured on the client clock and
therefore includes the network. Server identity is out of scope here (see the
endpoint identity work): this module records only what Metria itself
observed, plus safe endpoint fields.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..identity import SupportLevel
from ..models import RunSpec
from ..protocols import (
    CaptureRequest,
    InferenceBatch,
    InferenceRequest,
    RuntimeSession,
    SupportReport,
)
from .endpoint_http import (
    EndpointConnection,
    _env_name,
    bounded_retries,
    bounded_timeout,
)
from .endpoint_identity import (
    TokenizerCheck,
    check_declared,
    collect_claims,
    identity_evidence,
    validate_declared,
)
from .vllm_logprobs import TOP_LOGPROBS_CAPTURE, TOP_LOGPROBS_K

RUNTIME_NAME = "openai_endpoint"
# Flavors whose completion API returns token IDs and logprobs keyed by ID.
TOKEN_FLAVORS = frozenset({"vllm"})
FLAVORS = TOKEN_FLAVORS | {"generic"}
_RUNTIME_KEYS = frozenset(
    {"name", "flavor", "base_url_env", "api_key_env", "timeout_s", "max_retries"}
)
_DECODE_KEYS = ("max_tokens", "temperature", "seed")
_TOKENIZER_FILES = frozenset(
    {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
    }
)


def tokenizer_pins(value: Any) -> dict[str, str]:
    """Validate a tokenizer-only file-to-SHA-256 manifest."""

    if (
        not isinstance(value, Mapping)
        or "tokenizer.json" not in value
        or set(value) - _TOKENIZER_FILES
    ):
        raise ValueError(
            "model.tokenizer_files must pin tokenizer.json and only known "
            "tokenizer files: " + ", ".join(sorted(_TOKENIZER_FILES))
        )
    for digest in value.values():
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or set(digest) - set("0123456789abcdef")
        ):
            raise ValueError("tokenizer file pins must be lowercase SHA-256 digests")
    return dict(value)


def _runtime(spec: RunSpec) -> dict[str, Any]:
    runtime = dict(spec.runtime)
    if runtime.get("name") != RUNTIME_NAME:
        raise ValueError(f"runtime.name must be {RUNTIME_NAME!r}")
    if unknown := set(runtime) - _RUNTIME_KEYS - {"declared"}:
        raise ValueError(
            "unsupported endpoint runtime fields: " + ", ".join(sorted(unknown))
        )
    if runtime.get("flavor") not in FLAVORS:
        raise ValueError("runtime.flavor must be one of: " + ", ".join(sorted(FLAVORS)))
    _env_name(runtime.get("base_url_env"), "base_url_env")
    if runtime.get("api_key_env") is not None:
        _env_name(runtime["api_key_env"], "api_key_env")
    return {
        "name": RUNTIME_NAME,
        "flavor": runtime["flavor"],
        "base_url_env": runtime["base_url_env"],
        "api_key_env": runtime.get("api_key_env"),
        "timeout_s": bounded_timeout(runtime.get("timeout_s", 120)),
        "max_retries": bounded_retries(runtime.get("max_retries", 2)),
    }


def _model(spec: RunSpec, flavor: str) -> dict[str, Any]:
    model = spec.model
    if not isinstance(model.get("id"), str) or not model["id"]:
        raise ValueError("endpoint runs require model.id as served by the endpoint")
    if flavor not in TOKEN_FLAVORS:
        return {"id": model["id"]}
    if not isinstance(model.get("tokenizer_path"), str):
        raise ValueError(
            "token-ID endpoints require model.tokenizer_path and model.tokenizer_files"
        )
    files = tokenizer_pins(model.get("tokenizer_files"))
    return {
        "id": model["id"],
        "tokenizer_path": model["tokenizer_path"],
        "tokenizer_files": files,
    }


def _decode(scenario: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: scenario[key] for key in _DECODE_KEYS if key in scenario}
    tokens = result.get("max_tokens")
    if (
        isinstance(tokens, bool)
        or not isinstance(tokens, int)
        or not 1 <= tokens <= 32768
    ):
        raise ValueError("scenario.max_tokens must be an integer from 1 to 32768")
    return result


def load_tokenizer(path: str, files: Mapping[str, str]) -> Callable[[str], list[int]]:
    """Verify the pinned tokenizer files, then load ``tokenizer.json`` lazily."""

    import hashlib
    from pathlib import Path

    root = Path(path).expanduser().resolve()
    for name, digest in tokenizer_pins(files).items():
        hasher = hashlib.sha256()
        with (root / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                hasher.update(chunk)
        if hasher.hexdigest() != digest:
            raise RuntimeError(f"pinned tokenizer file {name} does not match")
    try:
        from tokenizers import Tokenizer  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "token-ID endpoints need the 'tokenizers' package to tokenize prompts locally"
        ) from exc
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    return lambda text: list(tokenizer.encode(text).ids)


def _token_id(key: Any) -> int:
    if not isinstance(key, str) or not key.startswith("token_id:"):
        raise RuntimeError("endpoint logprobs are not keyed by token ID")
    value = key.removeprefix("token_id:")
    if not value.isdigit():
        raise RuntimeError("endpoint logprob token ID is malformed")
    return int(value)


def _top_logprobs(
    choice: Mapping[str, Any], steps: int
) -> tuple[tuple[tuple[int, float], ...], ...]:
    """Keep exactly the K most likely tokens per step, highest first."""

    logprobs = choice.get("logprobs")
    rows = logprobs.get("top_logprobs") if isinstance(logprobs, Mapping) else None
    if not isinstance(rows, list) or len(rows) != steps:
        raise RuntimeError(
            "endpoint did not return top logprobs for every generated token"
        )
    result = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise RuntimeError("endpoint top-logprob step is malformed")
        pairs = []
        for key, value in row.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value > 0
            ):
                raise RuntimeError("endpoint logprobs must be finite and <= 0")
            pairs.append((_token_id(key), float(value)))
        pairs.sort(key=lambda pair: (-pair[1], pair[0]))
        if len(pairs) < TOP_LOGPROBS_K:
            raise RuntimeError(
                "endpoint returned fewer than the requested top-k logprobs"
            )
        result.append(tuple(pairs[:TOP_LOGPROBS_K]))
    return tuple(result)


def _token_ids(choice: Mapping[str, Any], prompt_ids: Sequence[int]) -> tuple[int, ...]:
    tokens = choice.get("token_ids")
    if not isinstance(tokens, list) or any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0
        for token in tokens
    ):
        raise RuntimeError("endpoint did not return generated token IDs")
    echoed = choice.get("prompt_token_ids")
    if echoed is not None and list(echoed) != list(prompt_ids):
        raise RuntimeError(
            "endpoint prompt token IDs differ from the locally tokenized prompt"
        )
    return tuple(tokens)


@dataclass
class EndpointSession:
    """One run's view of a remote server; ``close`` does not stop the server."""

    connection: EndpointConnection
    resolved: Mapping[str, Any]
    tokenize: Callable[[str], list[int]] | None
    claims: dict[str, dict[str, Any]] = field(default_factory=dict)
    tokenizer_check: TokenizerCheck | None = None
    invocations: list[dict[str, Any]] = field(default_factory=list)

    def _body(
        self, request: InferenceRequest, distributions: bool
    ) -> tuple[dict[str, Any], list[int] | None]:
        decode = {**self.resolved["decode"]}
        decode.update(
            {
                key: request.generation[key]
                for key in _DECODE_KEYS
                if key in request.generation
            }
        )
        body: dict[str, Any] = {
            "model": self.resolved["model"]["id"],
            "stream": False,
            **decode,
        }
        if self.tokenize is None:
            body["prompt"] = request.prompt
            return body, None
        prompt_ids = self.tokenize(request.prompt)
        body.update(prompt=prompt_ids, return_token_ids=True)
        if distributions:
            body.update(logprobs=TOP_LOGPROBS_K, return_tokens_as_token_ids=True)
        return body, prompt_ids

    def infer(
        self,
        requests: Sequence[InferenceRequest],
        capture: Sequence[CaptureRequest] = (),
    ) -> InferenceBatch:
        kinds = {item.kind for item in capture}
        if kinds - {"token_ids", TOP_LOGPROBS_CAPTURE}:
            raise RuntimeError("unsupported endpoint capture requested")
        if kinds and self.tokenize is None:
            raise RuntimeError("this endpoint flavor cannot return token captures")
        distributions = TOP_LOGPROBS_CAPTURE in kinds
        outputs: list[str] = []
        tokens: list[tuple[int, ...]] = []
        tops: list[Any] = []
        for request in requests:
            body, prompt_ids = self._body(request, distributions)
            if self.tokenizer_check is not None and prompt_ids is not None:
                self.tokenizer_check.check(request.prompt, prompt_ids)
            first_attempt = len(self.connection.attempts)
            payload, seconds = self.connection.request("POST", "/v1/completions", body)
            choices = payload.get("choices") if isinstance(payload, Mapping) else None
            if (
                not isinstance(choices, list)
                or len(choices) != 1
                or not isinstance(choices[0], Mapping)
            ):
                raise RuntimeError(
                    "endpoint completion response must contain one choice"
                )
            choice = choices[0]
            if not isinstance(choice.get("text"), str):
                raise RuntimeError("endpoint completion text is missing")
            outputs.append(choice["text"])
            row: dict[str, Any] = {
                "seconds": seconds,
                "finish_reason": choice.get("finish_reason"),
                "attempts": [dict(a) for a in self.connection.attempts[first_attempt:]],
            }
            if prompt_ids is not None:
                generated = _token_ids(choice, prompt_ids)
                tokens.append(generated)
                row.update(
                    prompt_tokens=len(prompt_ids), completion_tokens=len(generated)
                )
                if distributions:
                    tops.append(_top_logprobs(choice, len(generated)))
                    row["top_logprobs"] = {
                        "top_k": TOP_LOGPROBS_K,
                        "mode": "server_reported",
                        "qualification": "staged",
                    }
            self.invocations.append(row)
        captures: dict[str, Any] = {}
        if "token_ids" in kinds:
            captures["token_ids"] = tokens
        if distributions:
            captures[TOP_LOGPROBS_CAPTURE] = tops
        return InferenceBatch(
            outputs=tuple(outputs),
            captures=captures,
            metadata={"invocations": tuple(self.invocations[-len(requests) :])},
        )

    def reset(self, scope: str = "measurement") -> None:
        if scope != "measurement":
            raise RuntimeError(f"an endpoint session cannot reset {scope!r}")

    def close(self) -> None:
        """Nothing to release: the server's lifecycle is outside Metria."""


class EndpointAdapter:
    """Adapter for an OpenAI-compatible completions endpoint Metria does not own."""

    name = RUNTIME_NAME

    def probe(self, spec: RunSpec, environment: Mapping[str, Any]) -> SupportReport:
        del environment
        try:
            runtime = _runtime(spec)
            _model(spec, runtime["flavor"])
            _decode(spec.scenario)
            EndpointConnection.from_environment(
                runtime["base_url_env"],
                runtime["api_key_env"],
                timeout_s=runtime["timeout_s"],
                max_retries=runtime["max_retries"],
            )
        except ValueError as exc:
            return SupportReport(SupportLevel.UNSUPPORTED, reasons=(str(exc),))
        return SupportReport(
            SupportLevel.SUPPORTED,
            evidence={
                "flavor": runtime["flavor"],
                "network": "client_side_timing_includes_network",
            },
        )

    def probe_captures(
        self,
        spec: RunSpec,
        environment: Mapping[str, Any],
        capture: Sequence[CaptureRequest],
    ) -> SupportReport:
        del environment
        flavor = spec.runtime.get("flavor")
        kinds = {item.kind for item in capture}
        unknown = kinds - {"token_ids", TOP_LOGPROBS_CAPTURE}
        if unknown or (kinds and flavor not in TOKEN_FLAVORS):
            return SupportReport(
                SupportLevel.UNSUPPORTED,
                reasons=(
                    f"endpoint flavor {flavor!r} cannot provide: {', '.join(sorted(kinds))}",
                ),
            )
        if TOP_LOGPROBS_CAPTURE in kinds:
            return SupportReport(
                SupportLevel.EXPERIMENTAL,
                reasons=(
                    "server-reported top-k logprobs are staged pending qualification",
                ),
                evidence={"top_k": TOP_LOGPROBS_K},
            )
        return SupportReport(SupportLevel.SUPPORTED)

    def resolve(self, spec: RunSpec, environment: Mapping[str, Any]) -> dict[str, Any]:
        del environment
        runtime = _runtime(spec)
        return {
            "runtime": runtime,
            "model": _model(spec, runtime["flavor"]),
            "decode": _decode(spec.scenario),
            "declared": validate_declared(spec.runtime.get("declared")),
        }

    def launch(
        self, resolved: Mapping[str, Any], environment: Mapping[str, Any]
    ) -> EndpointSession:
        del environment
        runtime = resolved["runtime"]
        connection = EndpointConnection.from_environment(
            runtime["base_url_env"],
            runtime["api_key_env"],
            timeout_s=runtime["timeout_s"],
            max_retries=runtime["max_retries"],
        )
        model = resolved["model"]
        tokenize = None
        if runtime["flavor"] in TOKEN_FLAVORS:
            tokenize = load_tokenizer(model["tokenizer_path"], model["tokenizer_files"])
        claims = collect_claims(connection, runtime["flavor"], model["id"])
        check_declared(resolved["declared"], claims)
        check = TokenizerCheck(connection, model["id"]) if tokenize else None
        return EndpointSession(connection, resolved, tokenize, claims, check)

    def observe(self, session: RuntimeSession) -> dict[str, Any]:
        if not isinstance(session, EndpointSession):
            raise TypeError("EndpointAdapter can observe only its own sessions")
        runtime = session.resolved["runtime"]
        try:
            after: dict[str, Any] | None = collect_claims(
                session.connection, runtime["flavor"], session.resolved["model"]["id"]
            )
        except (RuntimeError, ValueError):
            after = None
        endpoint = session.connection.identity()
        tokenizer = (
            session.tokenizer_check.evidence() if session.tokenizer_check else None
        )
        return {
            "runtime": {"name": RUNTIME_NAME, "flavor": runtime["flavor"]},
            "identity": identity_evidence(session.claims, endpoint, tokenizer),
            "identity_authority": "claimed",
            "identity_snapshots": {
                "before": session.claims,
                "after": after,
                "stable": after == session.claims,
            },
            "declared": dict(session.resolved["declared"]),
            "endpoint": endpoint,
            "configured": {"decode": dict(session.resolved["decode"])},
            "invocations": [dict(row) for row in session.invocations],
        }
