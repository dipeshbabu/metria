"""STAGED vLLM decode-time distribution capture (raw top-k logprobs).

Status: STAGED. See docs/guides/staged-capabilities.md.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

# Staged decode-time distribution capture (see docs/guides/staged-capabilities.md).
TOP_LOGPROBS_CAPTURE = "token_top_logprobs"
TOP_LOGPROBS_K = 20
_RAW_LOGPROBS = "raw_logprobs"


def _logprobs_mode(llm: Any) -> str | None:
    """Return the engine's logprobs mode as reported by its model config."""

    engine = getattr(llm, "llm_engine", None)
    for config in (
        getattr(getattr(engine, "vllm_config", None), "model_config", None),
        getattr(engine, "model_config", None),
        getattr(llm, "model_config", None),
    ):
        mode = getattr(config, "logprobs_mode", None)
        if mode is not None:
            return str(getattr(mode, "value", mode))
    return None


def _top_logprobs(
    output: Any, tokens: Sequence[int], index: int
) -> tuple[tuple[tuple[int, float], ...], ...]:
    """Normalize vLLM per-token logprobs into rank-ordered top-k distributions.

    vLLM returns the top-k entries plus the sampled token when it ranks below
    k. Only ranks 1..k are kept so every step has a constant width.
    """

    steps = getattr(output, "logprobs", None)
    if isinstance(steps, (str, bytes)) or not isinstance(steps, Sequence):
        raise RuntimeError(f"vLLM response[{index}] has no per-token logprobs")
    if len(steps) != len(tokens):
        raise RuntimeError(
            f"vLLM response[{index}] logprobs do not cover every generated token"
        )
    distributions = []
    for step, entries in enumerate(steps):
        if not isinstance(entries, Mapping):
            raise RuntimeError(f"vLLM response[{index}] step {step} logprobs invalid")
        ranked = []
        for token, entry in entries.items():
            rank = getattr(entry, "rank", None)
            logprob = getattr(entry, "logprob", None)
            if (
                isinstance(token, bool)
                or not isinstance(token, int)
                or isinstance(rank, bool)
                or not isinstance(rank, int)
                or isinstance(logprob, bool)
                or not isinstance(logprob, (int, float))
            ):
                raise RuntimeError(
                    f"vLLM response[{index}] step {step} has malformed logprobs"
                )
            if 1 <= rank <= TOP_LOGPROBS_K:
                ranked.append((rank, token, float(logprob)))
        if not ranked:
            raise RuntimeError(
                f"vLLM response[{index}] step {step} has no top-k logprobs"
            )
        distributions.append(
            tuple((token, logprob) for _, token, logprob in sorted(ranked))
        )
    return tuple(distributions)
