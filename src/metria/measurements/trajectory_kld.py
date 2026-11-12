"""Staged decode-time distribution capture and trajectory KL divergence.

Status: STAGED. The native provider is the unqualified
``tools/qualification/staged/llamacpp-topk-capture.patch`` and no retained
native run backs this method yet. Results carry ``"qualification": "staged"``
and no policy target consumes them.

Trajectory agreement reports *where* sampled tokens diverge. This method
reports *how far* the model's next-token distribution moved before that point.
At decode step ``i`` the reference and candidate condition on identical context
only while their sampled prefixes agree, so each prompt is scored over steps
``0..first_divergence`` (inclusive) that both runs produced. Later steps have
different contexts and are excluded rather than compared.

Each step uses :func:`metria.fidelity.measurement_math.approximate_topk_kl`:
``KL(reference || candidate)`` over the union of visible top-k token IDs plus
one omitted-mass bucket. It is an approximation and is labeled as one.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .._numbers import is_finite
from ..fidelity.measurement_math import approximate_topk_kl
from ..models import (
    MetricDefinition,
    MetricDirection,
    MetricSample,
    MetricSummary,
    RunRecord,
)
from ..protocols import (
    CaptureRequest,
    InferenceRequest,
    MeasurementResult,
    RuntimeSession,
)
from .trajectory import (
    _capture_rows,
    _first_divergence,
    _prompt_rows,
    _sha256_text,
    _token_batches,
)

QUALIFICATION = "staged"
TOP_LOGPROBS_CAPTURE = "token_top_logprobs"
_CAPTURE_SCHEMA = "metria.trajectory_distribution_capture.v1"
_CAPTURE_METHOD = "metria.decode_time_top_logprobs"
_CAPTURE_VERSION = "0.1.0"
_COMPARISON_SCHEMA = "metria.trajectory_kld.v1"
_COMPARISON_METHOD = "metria.trajectory_kld"
_COMPARISON_VERSION = "0.1.0"
_LOG_FLOOR = -30.0
_TRAJECTORY_SCHEMA = "metria.trajectory_capture.v1"
_TRAJECTORY_METHOD = "kv_fidelity.decode_time_trajectory"
_TRAJECTORY_VERSION = "0.3.4"


def _distribution(value: Any, *, label: str) -> dict[int, float]:
    """Convert one retained ``[[token_id, logprob], ...]`` row into a mapping."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError(f"{label} must be a non-empty sequence of pairs")
    distribution: dict[int, float] = {}
    for item in value:
        if isinstance(item, (str, bytes)) or not isinstance(item, Sequence):
            raise ValueError(f"{label} entries must be [token_id, logprob] pairs")
        if len(item) != 2:
            raise ValueError(f"{label} entries must be [token_id, logprob] pairs")
        token, logprob = item
        if isinstance(token, bool) or not isinstance(token, int) or token < 0:
            raise ValueError(f"{label} requires non-negative integer token IDs")
        if token in distribution:
            raise ValueError(f"{label} repeats token ID {token}")
        if (
            isinstance(logprob, bool)
            or not isinstance(logprob, (int, float))
            or not is_finite(logprob)
            or logprob > 0.0
        ):
            raise ValueError(f"{label} log probabilities must be finite and <= 0")
        distribution[token] = float(logprob)
    return distribution


def _distribution_batches(
    result: Any, *, token_batches: Sequence[Sequence[int]]
) -> tuple[tuple[tuple[tuple[int, float], ...], ...], ...]:
    """Validate runtime distribution capture against the token capture shape."""

    if isinstance(result, (str, bytes)) or not isinstance(result, Sequence):
        raise RuntimeError(f"runtime {TOP_LOGPROBS_CAPTURE} capture must be a sequence")
    if len(result) != len(token_batches):
        raise RuntimeError(
            f"runtime {TOP_LOGPROBS_CAPTURE} capture count does not match prompts"
        )
    batches = []
    for index, (steps, tokens) in enumerate(zip(result, token_batches, strict=True)):
        if isinstance(steps, (str, bytes)) or not isinstance(steps, Sequence):
            raise RuntimeError(f"{TOP_LOGPROBS_CAPTURE}[{index}] must be a sequence")
        if len(steps) != len(tokens):
            raise RuntimeError(
                f"{TOP_LOGPROBS_CAPTURE}[{index}] does not cover every decode step"
            )
        rows = []
        for step, raw in enumerate(steps):
            try:
                parsed = _distribution(raw, label=f"prompt {index} step {step}")
            except ValueError as exc:
                raise RuntimeError(str(exc)) from None
            rows.append(tuple(parsed.items()))
        batches.append(tuple(rows))
    return tuple(batches)


class TokenDistributionProtocol:
    """STAGED: capture sampled tokens plus top-k next-token distributions."""

    name = _CAPTURE_METHOD
    version = _CAPTURE_VERSION

    def requirements(self, config: Mapping[str, Any]) -> tuple[CaptureRequest, ...]:
        """Require token IDs and the staged distribution capture."""

        _prompt_rows(config)
        return (
            CaptureRequest(kind="token_ids"),
            CaptureRequest(kind=TOP_LOGPROBS_CAPTURE),
        )

    def execute(
        self,
        session: RuntimeSession,
        scenario: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> MeasurementResult:
        """Run prompts and retain trajectories with per-step distributions."""

        del scenario
        rows = _prompt_rows(config)
        batch = session.infer(
            tuple(
                InferenceRequest(prompt=row["prompt"], generation=row["generation"])
                for row in rows
            ),
            capture=self.requirements(config),
        )
        for kind in ("token_ids", TOP_LOGPROBS_CAPTURE):
            if kind not in batch.captures:
                raise RuntimeError(
                    f"runtime did not return the required {kind} capture"
                )
        tokens = _token_batches(batch.captures["token_ids"], expected=len(rows))
        distributions = _distribution_batches(
            batch.captures[TOP_LOGPROBS_CAPTURE], token_batches=tokens
        )
        evidence_rows = [
            {
                "id": row["id"],
                "category": row["category"],
                "prompt_sha256": _sha256_text(row["prompt"]),
                "token_ids": row_tokens,
                "token_count": len(row_tokens),
                "top_logprobs": tuple(
                    tuple([token, logprob] for token, logprob in step)
                    for step in row_distributions
                ),
            }
            for row, row_tokens, row_distributions in zip(
                rows, tokens, distributions, strict=True
            )
        ]
        steps = sum(len(row_tokens) for row_tokens in tokens)
        return MeasurementResult(
            metrics={
                "distribution_capture_steps": MetricSummary(
                    definition=MetricDefinition(
                        name="distribution_capture_steps",
                        unit="tokens",
                        direction=MetricDirection.DESCRIPTIVE,
                        method=_CAPTURE_METHOD,
                        version=_CAPTURE_VERSION,
                    ),
                    value=float(steps),
                    aggregation="sum",
                    coverage=1.0,
                )
            },
            evidence={
                "schema": _CAPTURE_SCHEMA,
                "method": _CAPTURE_METHOD,
                "method_version": _CAPTURE_VERSION,
                "qualification": QUALIFICATION,
                "n_prompts": len(rows),
                "prompts": tuple(evidence_rows),
            },
        )


def _validated_rows(result: MeasurementResult) -> tuple[Mapping[str, Any], ...]:
    """Validate distribution capture evidence through the trajectory checks."""

    evidence = result.evidence
    if (
        evidence.get("schema"),
        evidence.get("method"),
        evidence.get("method_version"),
    ) != (_CAPTURE_SCHEMA, _CAPTURE_METHOD, _CAPTURE_VERSION):
        raise ValueError("measurement result is not staged distribution capture")
    rows = _capture_rows(
        MeasurementResult(
            evidence={
                **evidence,
                "schema": _TRAJECTORY_SCHEMA,
                "method": _TRAJECTORY_METHOD,
                "method_version": _TRAJECTORY_VERSION,
            }
        )
    )
    width: int | None = None
    for row in rows:
        steps = row.get("top_logprobs")
        if (
            isinstance(steps, (str, bytes))
            or not isinstance(steps, Sequence)
            or len(steps) != row["token_count"]
        ):
            raise ValueError(
                f"distribution evidence for prompt {row['id']!r} does not cover "
                "every decode step"
            )
        for step, raw in enumerate(steps):
            parsed = _distribution(raw, label=f"prompt {row['id']!r} step {step}")
            if width is None:
                width = len(parsed)
            elif len(parsed) != width:
                raise ValueError("distribution evidence top-k width is inconsistent")
    return rows


def _top_k(rows: Sequence[Mapping[str, Any]]) -> int | None:
    for row in rows:
        if row["top_logprobs"]:
            return len(row["top_logprobs"][0])
    return None


def _score_prompt(
    reference_row: Mapping[str, Any], candidate_row: Mapping[str, Any]
) -> tuple[dict[str, Any], list[float]]:
    """Score one prompt over the decode steps where both contexts agree."""

    prompt_id = str(reference_row["id"])
    if reference_row["prompt_sha256"] != candidate_row["prompt_sha256"]:
        raise ValueError(f"prompt fingerprint differs for prompt id {prompt_id!r}")
    reference_tokens = tuple(reference_row["token_ids"])
    candidate_tokens = tuple(candidate_row["token_ids"])
    first_divergence, _ = _first_divergence(reference_tokens, candidate_tokens)
    shared = min(len(reference_tokens), len(candidate_tokens))
    scored = shared if first_divergence is None else min(first_divergence + 1, shared)
    values = [
        approximate_topk_kl(
            _distribution(reference_row["top_logprobs"][step], label="reference"),
            _distribution(candidate_row["top_logprobs"][step], label="candidate"),
            log_floor=_LOG_FLOOR,
        )
        for step in range(scored)
    ]
    row = {
        "id": prompt_id,
        "prompt_sha256": reference_row["prompt_sha256"],
        "first_divergence": first_divergence,
        "scored_steps": scored,
        "mean_kld": math.fsum(values) / scored if scored else None,
        "max_kld": max(values) if values else None,
    }
    return row, values


def _kld_metric(
    name: str,
    value: float,
    aggregation: str,
    samples: Sequence[MetricSample],
    coverage: float,
) -> MetricSummary:
    return MetricSummary(
        definition=MetricDefinition(
            name=name,
            unit="nats",
            direction=MetricDirection.LOWER_IS_BETTER,
            method=_COMPARISON_METHOD,
            version=_COMPARISON_VERSION,
        ),
        value=value,
        samples=tuple(samples),
        aggregation=aggregation,
        coverage=coverage,
    )


def compare_trajectory_distributions(
    reference: MeasurementResult,
    candidate: MeasurementResult,
) -> MeasurementResult:
    """STAGED: KL(reference || candidate) over shared-context decode steps."""

    reference_rows = _validated_rows(reference)
    candidate_rows = _validated_rows(candidate)
    candidate_by_id = {str(row["id"]): row for row in candidate_rows}
    if {str(row["id"]) for row in reference_rows} != set(candidate_by_id):
        raise ValueError("distribution capture prompt sets differ")
    reference_k, candidate_k = _top_k(reference_rows), _top_k(candidate_rows)
    if None not in (reference_k, candidate_k) and reference_k != candidate_k:
        raise ValueError(
            "reference and candidate top-k widths differ; the estimator "
            "requires the same capture width"
        )

    per_prompt: list[dict[str, Any]] = []
    samples: list[MetricSample] = []
    step_values: list[float] = []
    for reference_row in reference_rows:
        row, values = _score_prompt(
            reference_row, candidate_by_id[str(reference_row["id"])]
        )
        per_prompt.append(row)
        step_values.extend(values)
        if values:
            samples.append(
                MetricSample(
                    row["mean_kld"],
                    metadata={"prompt_id": row["id"], "scored_steps": len(values)},
                )
            )
    if not step_values:
        raise RuntimeError(
            "no decode step shares context between reference and candidate; "
            "trajectory KL divergence cannot be scored"
        )

    n_prompts = len(reference_rows)
    coverage = len(samples) / n_prompts
    notes = []
    if coverage < 1.0:
        notes.append(
            f"{n_prompts - len(samples)}/{n_prompts} prompts had no "
            "shared-context decode step and are excluded"
        )
    return MeasurementResult(
        metrics={
            "trajectory_kld_mean": _kld_metric(
                "trajectory_kld_mean",
                math.fsum(step_values) / len(step_values),
                "mean_over_shared_context_steps",
                samples,
                coverage,
            ),
            "trajectory_kld_max": _kld_metric(
                "trajectory_kld_max",
                max(step_values),
                "max_over_shared_context_steps",
                samples,
                coverage,
            ),
        },
        evidence={
            "schema": _COMPARISON_SCHEMA,
            "method": _COMPARISON_METHOD,
            "method_version": _COMPARISON_VERSION,
            "qualification": QUALIFICATION,
            "estimator": {
                "name": "approximate_topk_kl",
                "direction": "KL(reference || candidate)",
                "top_k": reference_k if reference_k is not None else candidate_k,
                "log_floor": _LOG_FLOOR,
                "omitted_mass_bucket": True,
            },
            "scope": "steps 0..first_divergence produced by both runs",
            "n_prompts": n_prompts,
            "scored_prompts": len(samples),
            "scored_steps": len(step_values),
            "per_prompt": tuple(per_prompt),
            "notes": tuple(notes),
        },
    )


class TrajectoryDivergenceAnalysis:
    """STAGED: derive trajectory KL divergence from two run-local captures."""

    name = _COMPARISON_METHOD
    version = _COMPARISON_VERSION

    def analyze(self, left: RunRecord, right: RunRecord) -> MeasurementResult:
        """Treat the left run as reference and the right run as candidate."""

        return compare_trajectory_distributions(
            self._capture_result(left), self._capture_result(right)
        )

    @staticmethod
    def _capture_result(record: RunRecord) -> MeasurementResult:
        measurements = record.evidence.get("measurements")
        if not isinstance(measurements, Mapping):
            raise ValueError(
                f"run {record.run_id!r} does not retain measurement evidence"
            )
        raw = measurements.get(TokenDistributionProtocol.name)
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"run {record.run_id!r} does not contain distribution capture evidence"
            )
        return MeasurementResult(evidence=raw)
