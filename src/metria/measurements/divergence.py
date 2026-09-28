"""Deterministic summaries of paired, identity-checked token trajectories."""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


def summarize_divergence(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize validated comparison rows without token or prompt contents.

    Positions are zero-based. Ranking favors the lowest prefix fraction, then
    earliest divergence, then prompt ID. Empty captures are unavailable evidence,
    even though the legacy trajectory score penalizes unilateral empty outputs.
    """
    complete = [
        row for row in rows if row["reference_steps"] and row["candidate_steps"]
    ]
    unavailable = len(rows) - len(complete)
    diverged = [row for row in complete if not row["matched"]]
    positions = [row["first_divergence"] for row in diverged]
    distribution = Counter(positions)
    categories: dict[str, list[Mapping[str, Any]]] = {}
    for row in complete:
        if row.get("category") is not None:
            categories.setdefault(row["category"], []).append(row)
    ranked = sorted(
        diverged,
        key=lambda row: (
            row["prefix_agreement_steps"]
            / max(row["reference_steps"], row["candidate_steps"]),
            row["first_divergence"],
            row["id"],
        ),
    )
    total = len(complete)
    return {
        "schema": "metria.trajectory_divergence.v1",
        "status": "complete" if not unavailable else "insufficient_evidence",
        "n_prompts": len(rows),
        "compared_prompts": total,
        "unavailable_prompts": unavailable,
        "exact_matches": total - len(diverged),
        "exact_match_rate": (total - len(diverged)) / total if total else None,
        "diverged_prompts": len(diverged),
        "divergence_rate": len(diverged) / total if total else None,
        "mean_prefix_steps": (
            sum(row["prefix_agreement_steps"] for row in complete) / total
            if total
            else None
        ),
        "first_divergence_positions": [
            {"position": position, "count": count}
            for position, count in sorted(distribution.items())
        ],
        "earliest_first_divergence": min(positions) if positions else None,
        "median_first_divergence": statistics.median(positions) if positions else None,
        "position_origin": 0,
        "length_mismatches": sum(
            row["reference_steps"] != row["candidate_steps"] for row in complete
        ),
        "by_category": [
            {
                "category": category,
                "n_prompts": len(members),
                "diverged_prompts": sum(not row["matched"] for row in members),
                "divergence_rate": sum(not row["matched"] for row in members)
                / len(members),
            }
            for category, members in sorted(categories.items())
        ],
        "uncategorized_prompts": sum(row.get("category") is None for row in complete),
        "top_limit": 10,
        "ranking": "prefix_fraction_ascending,first_divergence_ascending,id_ascending",
        "most_divergent": [dict(row) for row in ranked[:10]],
    }
