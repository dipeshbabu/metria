"""STAGED: top-k distributions retained alongside prefix-workload trajectories.

A prefix workload that sets ``"staged_distributions": true`` also requests the
experimental ``token_top_logprobs`` capture on every call. Measured-trial
distributions are retained under the trajectory evidence's ``distributions``
key in the staged distribution-capture schema, so the trajectory evidence and
its qualified analysis stay unchanged while ``metria.trajectory_kld`` can score
the same decode steps.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..protocols import CaptureRequest
from .trajectory_kld import (
    _CAPTURE_METHOD,
    _CAPTURE_SCHEMA,
    _CAPTURE_VERSION,
    QUALIFICATION,
    TOP_LOGPROBS_CAPTURE,
    _distribution_batches,
)

STAGED_DISTRIBUTIONS = "staged_distributions"
DISTRIBUTIONS_KEY = "distributions"
DISTRIBUTION_CAPTURE = CaptureRequest(kind=TOP_LOGPROBS_CAPTURE)


def staged_distributions(config: Mapping[str, Any]) -> bool:
    """Return the validated opt-in flag; absent means no distribution capture."""

    value = config.get(STAGED_DISTRIBUTIONS, False)
    if not isinstance(value, bool):
        raise TypeError(f"{STAGED_DISTRIBUTIONS} must be a boolean")
    return value


def distribution_evidence(
    rows: Sequence[Mapping[str, Any]], distributions: Sequence[Any]
) -> dict[str, Any]:
    """Pair measured trajectory rows with their captured per-step distributions."""

    batches = _distribution_batches(
        distributions, token_batches=[tuple(row["token_ids"]) for row in rows]
    )
    prompts = tuple(
        {
            **row,
            "top_logprobs": tuple(
                tuple([token, logprob] for token, logprob in step) for step in steps
            ),
        }
        for row, steps in zip(rows, batches, strict=True)
    )
    return {
        "schema": _CAPTURE_SCHEMA,
        "method": _CAPTURE_METHOD,
        "method_version": _CAPTURE_VERSION,
        "qualification": QUALIFICATION,
        "n_prompts": len(prompts),
        "prompts": prompts,
    }
