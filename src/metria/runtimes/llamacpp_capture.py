"""Strict readers for the qualified local llama.cpp capture provider."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .._numbers import is_finite
from ..recipes import _unique_json_object

CAPTURE_SCHEMA = "metria.llamacpp_capture.v1"
_INTEGER_FIELDS = frozenset({"context", "threads", "threads_batch", "vocab_size"})


def read_runtime_capture(path: Path) -> Mapping[str, Any] | None:
    """Read bounded runtime facts; older token-only providers remain explicit."""
    try:
        if path.stat().st_size > 4096:
            raise ValueError("llama.cpp runtime capture exceeds the supported size")
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    value = json.loads(text, object_pairs_hook=_unique_json_object)
    expected = {"schema", "chat_template_applied", *_INTEGER_FIELDS}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("invalid llama.cpp runtime capture fields")
    if value["schema"] != CAPTURE_SCHEMA:
        raise ValueError("unsupported llama.cpp runtime capture schema")
    for field in _INTEGER_FIELDS:
        item = value[field]
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(
                f"llama.cpp runtime capture {field} must be a positive integer"
            )
    if not isinstance(value["chat_template_applied"], bool):
        raise ValueError(
            "llama.cpp runtime capture chat_template_applied must be boolean"
        )
    return value


TopLogprobs = tuple[tuple[int, float], ...]
# Staged: top-k capture comes from tools/qualification/staged/llamacpp-topk-capture.patch
# and is not part of any qualified provider.
MAX_TOP_K = 256
_MASS_TOLERANCE = 1e-6


def _reject_constant(value: str) -> Any:
    raise ValueError(f"llama.cpp capture contains non-finite number {value}")


def _top_logprobs(value: Any, *, top_k: int, step: int) -> TopLogprobs:
    """Validate one full-vocabulary top-k distribution written by the provider."""
    label = f"llama.cpp top-k capture step {step}"
    if not isinstance(value, list) or not 1 <= len(value) <= top_k:
        raise ValueError(f"{label} must list between 1 and {top_k} entries")
    entries: list[tuple[int, float]] = []
    seen: set[int] = set()
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"{label} entries must be [token_id, logprob] pairs")
        token, logprob = item
        if isinstance(token, bool) or not isinstance(token, int) or token < 0:
            raise ValueError(f"{label} requires non-negative integer token IDs")
        if token in seen:
            raise ValueError(f"{label} repeats token ID {token}")
        if (
            isinstance(logprob, bool)
            or not isinstance(logprob, (int, float))
            or not is_finite(logprob)
            or logprob > 0.0
        ):
            raise ValueError(f"{label} log probabilities must be finite and <= 0")
        if entries and logprob > entries[-1][1]:
            raise ValueError(f"{label} entries must be sorted by probability")
        seen.add(token)
        entries.append((token, float(logprob)))
    if math.fsum(math.exp(logprob) for _, logprob in entries) > 1.0 + _MASS_TOLERANCE:
        raise ValueError(f"{label} probability mass exceeds one")
    return tuple(entries)


def read_token_capture(
    path: Path, *, top_k: int | None = None
) -> tuple[list[int], list[TopLogprobs] | None]:
    """Read sampled token IDs and, when requested, staged top-k distributions.

    Without ``top_k`` the reader accepts only the qualified token-ID ABI. With
    ``top_k`` every step must carry a distribution of one consistent width.
    """
    if top_k is not None and (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or not 1 <= top_k <= MAX_TOP_K
    ):
        raise ValueError(f"top_k must be an integer from 1 to {MAX_TOP_K}")
    tokens: list[int] = []
    distributions: list[TopLogprobs] | None = [] if top_k is not None else None
    allowed = {"step", "token_id"} | ({"top_logprobs"} if top_k else set())
    try:
        stream = path.open(encoding="utf-8")
    except FileNotFoundError:
        return tokens, distributions
    with stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(
                line,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_constant,
            )
            if not isinstance(row, dict) or not set(row) <= allowed:
                raise ValueError("invalid llama.cpp token capture fields")
            token = row.get("token_id")
            if isinstance(token, bool) or not isinstance(token, int) or token < 0:
                raise ValueError(
                    "llama.cpp token capture requires non-negative integer IDs"
                )
            if "step" in row:
                step = row["step"]
                if (
                    isinstance(step, bool)
                    or not isinstance(step, int)
                    or step != len(tokens)
                ):
                    raise ValueError(
                        "llama.cpp token capture steps must be contiguous from zero"
                    )
            if distributions is not None:
                assert top_k is not None
                if "top_logprobs" not in row:
                    raise ValueError(
                        "llama.cpp top-k capture is missing for step "
                        f"{len(tokens)}; the provider lacks the staged top-k patch"
                    )
                distribution = _top_logprobs(
                    row["top_logprobs"], top_k=top_k, step=len(tokens)
                )
                if distributions and len(distribution) != len(distributions[0]):
                    raise ValueError("llama.cpp top-k capture width changed mid-run")
                distributions.append(distribution)
            tokens.append(token)
    return tokens, distributions


def read_token_trajectory(path: Path) -> list[int]:
    """Read sampled token IDs without coercing malformed capture data."""
    return read_token_capture(path)[0]
