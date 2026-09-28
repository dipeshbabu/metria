"""Shared chunking, alignment, validation, and top-k KLD aggregation."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .contracts import BackendCapabilityError, _TopKKLDMetrics


def full_token_chunks(
    token_ids: Sequence[int], *, chunk_len: int, max_chunks: int
) -> list[list[int]]:
    """Return complete, non-overlapping token chunks up to ``max_chunks``.

    A final partial chunk is deliberately ignored because backend KLD runs
    compare fixed-size contexts. An input containing exactly ``chunk_len``
    tokens therefore yields one chunk.
    """
    if chunk_len <= 0:
        raise ValueError("chunk_len must be positive")
    if max_chunks <= 0:
        return []

    stop = len(token_ids) - chunk_len + 1
    return [
        list(token_ids[start : start + chunk_len])
        for start in range(0, stop, chunk_len)
    ][:max_chunks]


def approximate_topk_kl(
    reference_logprobs: dict[int, float],
    candidate_logprobs: dict[int, float],
    *,
    log_floor: float = -30.0,
) -> float:
    """Return a normalized top-k KL estimate with an omitted-mass bucket.

    Native vLLM and SGLang APIs expose only top-k log probabilities. Treating
    their partial sum as a full distribution is not KL divergence and can
    produce misleading cross-backend values. This helper aligns the union of
    visible token IDs, assigns a small floor to tokens absent on one side,
    adds one bucket for all omitted vocabulary mass, normalizes both vectors,
    and computes ``KL(reference || candidate)``.

    It remains an approximation and callers must label it as such.
    """
    if not reference_logprobs or not candidate_logprobs:
        raise ValueError("both top-k distributions must be non-empty")
    if not math.isfinite(log_floor) or log_floor > 0.0:
        raise ValueError("log_floor must be a finite, non-positive value")

    for label, logprobs in (
        ("reference", reference_logprobs),
        ("candidate", candidate_logprobs),
    ):
        for token_id, logprob in logprobs.items():
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise ValueError(f"{label} token ID must be an integer")
            try:
                finite = math.isfinite(logprob)
            except TypeError as exc:
                raise ValueError(
                    f"{label} log probability for token {token_id} must be numeric"
                ) from exc
            if not finite:
                raise ValueError(
                    f"{label} log probability for token {token_id} must be finite"
                )
            if logprob > 0.0:
                raise ValueError(
                    f"{label} log probability for token {token_id} must be <= 0"
                )

    floor = math.exp(log_floor)
    token_ids = sorted(set(reference_logprobs) | set(candidate_logprobs))
    p = [math.exp(reference_logprobs.get(tid, log_floor)) for tid in token_ids]
    q = [math.exp(candidate_logprobs.get(tid, log_floor)) for tid in token_ids]
    p.append(max(1.0 - sum(p), floor))
    q.append(max(1.0 - sum(q), floor))

    p_total = sum(p)
    q_total = sum(q)
    p = [value / p_total for value in p]
    q = [value / q_total for value in q]
    value = sum(pi * math.log(pi / qi) for pi, qi in zip(p, q, strict=True) if pi > 0.0)
    if not math.isfinite(value):
        raise ValueError("top-k KL estimate is non-finite")
    return max(value, 0.0)


def aggregate_topk_kld(
    reference_chunks: Sequence[Sequence[dict[int, float]]],
    candidate_chunks: Sequence[Sequence[dict[int, float]]],
    *,
    backend_name: str,
    log_floor: float = -30.0,
) -> _TopKKLDMetrics:
    """Validate paired top-k responses and aggregate their KLD diagnostics.

    Top-k token sets may differ and are aligned by ``approximate_topk_kl``.
    Chunk and position counts must match exactly. A position omitted by both
    responses is treated as explicitly unscored and surfaced in coverage
    metadata; one-sided omissions are alignment failures.
    """
    if len(reference_chunks) != len(candidate_chunks):
        raise BackendCapabilityError(
            f"{backend_name} KLD response chunk count mismatch: reference "
            f"returned {len(reference_chunks)}, candidate returned "
            f"{len(candidate_chunks)}. The responses cannot be aligned safely."
        )

    total_kl = 0.0
    n_positions_total = 0
    n_positions_scored = 0
    n_positions_skipped = 0
    sq_dp_sum = 0.0
    n_dp = 0
    same_topp_hits = 0

    for chunk_index, (reference_chunk, candidate_chunk) in enumerate(
        zip(reference_chunks, candidate_chunks, strict=True)
    ):
        if len(reference_chunk) != len(candidate_chunk):
            raise BackendCapabilityError(
                f"{backend_name} KLD response position count mismatch in chunk "
                f"{chunk_index}: reference returned {len(reference_chunk)}, "
                f"candidate returned {len(candidate_chunk)}. The responses "
                "cannot be aligned safely."
            )

        for position_index, (reference_position, candidate_position) in enumerate(
            zip(reference_chunk, candidate_chunk, strict=True)
        ):
            n_positions_total += 1
            if not reference_position and not candidate_position:
                n_positions_skipped += 1
                continue
            if not reference_position or not candidate_position:
                missing_side = "reference" if not reference_position else "candidate"
                raise BackendCapabilityError(
                    f"{backend_name} KLD log-probability availability mismatch "
                    f"at chunk {chunk_index}, position {position_index}: "
                    f"{missing_side} response is empty."
                )

            try:
                position_kld = approximate_topk_kl(
                    reference_position,
                    candidate_position,
                    log_floor=log_floor,
                )
            except ValueError as exc:
                raise BackendCapabilityError(
                    f"{backend_name} returned invalid KLD log probabilities at "
                    f"chunk {chunk_index}, position {position_index}: {exc}"
                ) from exc
            if not math.isfinite(position_kld):
                raise BackendCapabilityError(
                    f"{backend_name} produced a non-finite per-position KLD at "
                    f"chunk {chunk_index}, position {position_index}."
                )

            n_positions_scored += 1
            total_kl += position_kld
            for token_id, reference_logprob in reference_position.items():
                probability = math.exp(reference_logprob)
                if probability > 1e-9:
                    candidate_logprob = candidate_position.get(token_id, log_floor)
                    relative_delta = (
                        math.exp(candidate_logprob) - probability
                    ) / probability
                    sq_dp_sum += relative_delta**2
                    n_dp += 1

            reference_top = max(reference_position.items(), key=lambda item: item[1])[0]
            candidate_top = max(candidate_position.items(), key=lambda item: item[1])[0]
            same_topp_hits += int(reference_top == candidate_top)

    if n_positions_scored == 0:
        raise BackendCapabilityError(
            f"{backend_name} KLD response contained zero usable positions. "
            "The engine may not support prompt log probabilities or its "
            "response schema may have changed."
        )

    mean_kld = total_kl / n_positions_scored
    rms_dp_pct = 100.0 * math.sqrt(sq_dp_sum / n_dp) if n_dp else None
    same_topp_pct = 100.0 * same_topp_hits / n_positions_scored
    for metric_name, metric in (
        ("mean KLD", mean_kld),
        ("RMS distribution delta", rms_dp_pct),
        ("same-top-token percentage", same_topp_pct),
    ):
        if metric is not None and not math.isfinite(metric):
            raise BackendCapabilityError(
                f"{backend_name} produced a non-finite final {metric_name}."
            )

    return _TopKKLDMetrics(
        mean_kld=mean_kld,
        rms_dp_pct=rms_dp_pct,
        same_topp_pct=same_topp_pct,
        n_positions_total=n_positions_total,
        n_positions_scored=n_positions_scored,
        n_positions_skipped=n_positions_skipped,
    )
