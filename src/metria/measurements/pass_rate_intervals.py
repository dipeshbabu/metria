"""95% confidence intervals for task-check pass rates.

Each retained check outcome is treated as one Bernoulli trial. Checks on the
same prompt, or on the same prompt under different seeds, are not strictly
independent, so the intervals are approximate and say so in evidence. Wilson
score intervals bound each pass rate; Newcombe's hybrid score method bounds the
candidate-minus-reference difference.
"""

from __future__ import annotations

import math

Z_95 = 1.959963984540054
METHOD = "metria.task_pass_rate_interval"
VERSION = "1"
ASSUMPTION = (
    "each retained check outcome is treated as an independent Bernoulli trial; "
    "checks sharing a prompt or seed are correlated, so intervals are approximate"
)


def wilson(passed: int, total: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for ``passed / total``."""

    if total <= 0 or not 0 <= passed <= total:
        raise ValueError("pass counts must satisfy 0 <= passed <= total and total > 0")
    rate = passed / total
    denominator = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
    half /= denominator
    lower = 0.0 if passed == 0 else max(0.0, centre - half)
    upper = 1.0 if passed == total else min(1.0, centre + half)
    return lower, upper


def newcombe(
    reference_passed: int,
    reference_total: int,
    candidate_passed: int,
    candidate_total: int,
) -> tuple[float, float]:
    """Hybrid score interval for candidate rate minus reference rate."""

    p1 = reference_passed / reference_total
    p2 = candidate_passed / candidate_total
    l1, u1 = wilson(reference_passed, reference_total)
    l2, u2 = wilson(candidate_passed, candidate_total)
    delta = p2 - p1
    lower = delta - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2)
    upper = delta + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)
    return max(-1.0, lower), min(1.0, upper)


def intervals(
    reference_passed: int, candidate_passed: int, total: int
) -> tuple[dict[str, object], dict[str, float]]:
    """Evidence and metric values for one pair of equal-sized check sets."""

    reference = wilson(reference_passed, total)
    candidate = wilson(candidate_passed, total)
    delta = newcombe(reference_passed, total, candidate_passed, total)
    evidence = {
        "method": METHOD,
        "method_version": VERSION,
        "confidence": 0.95,
        "assumption": ASSUMPTION,
        "reference_pass_rate": list(reference),
        "candidate_pass_rate": list(candidate),
        "pass_rate_delta": list(delta),
    }
    values = {
        "candidate_task_pass_rate_lower": candidate[0],
        "task_pass_rate_delta_lower": delta[0],
        "task_pass_rate_delta_upper": delta[1],
    }
    return evidence, values
