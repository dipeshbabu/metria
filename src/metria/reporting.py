"""Deterministic human-readable projections of canonical verification results."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from typing import Any

from .measurements import TrajectoryAgreementAnalysis
from .policies import render_policy_evaluation
from .verification_schema import VERIFICATION_ROLES as _ROLES
from .verification_schema import VERIFICATION_SCOPE


def render_verification(manifest: Mapping[str, Any]) -> str:
    """Render a concise report without raw prompts or configuration values."""
    lines = [
        "# Metria Verification",
        "",
        f"**Verdict: {manifest['verdict']}**",
        "",
        f"Recipe: `{manifest['recipe_digest']}`",
        (
            "Scope: synthetic fixture; no real runtime or model qualification"
            if manifest.get("fixture_only") is True
            else "Scope: local llama.cpp CPU thread comparison"
            if manifest.get("scope") == VERIFICATION_SCOPE
            else "Scope: unrecognized verification contract"
        ),
        "",
        "## Change:",
        f"  CPU threads: {manifest['change']['reference_threads']} -> {manifest['change']['candidate_threads']}",
        "",
        "## Evidence:",
    ]
    for role in _ROLES:
        record = manifest["records"][role]
        lines.append(f"  {role}: {record['status']} ({record['path']})")
        facts = record["observed"]
        lines.append(
            f"    Observed threads: {facts['threads']}; context: {facts['context']}"
        )
        for gap in record["evidence_gaps"]:
            lines.append(f"    {gap}")
    if "lifecycle" in manifest:
        lines.append(f"  Lifecycle: {manifest['lifecycle']['status']}")
    for control in manifest.get("controls", ()):
        lines.append(f"  {control['dimension']}: {control['status']}")
    lines.extend(
        (
            "",
            "## Comparison:",
            f"  {manifest.get('comparison_status', 'VALID' if manifest['comparison']['compatible'] else 'NOT_COMPARABLE')}",
        )
    )
    for issue in manifest["comparison"]["issues"]:
        lines.append(f"  {issue['dimension']}: {issue['reason']}")
    for issue in manifest["comparison"]["waived_differences"]:
        lines.append(
            f"  Waived difference: {issue['dimension']} (rationale retained by digest)"
        )
    lines.extend(("", "## Impact:"))
    for analysis in manifest["analyses"]:
        lines.append(f"  {analysis['name']}: {analysis['status']}")
        metrics = analysis["metrics"]
        for key, error in sorted(analysis.get("metric_errors", {}).items()):
            lines.append(f"    {key}: invalid metric evidence ({error['error_type']})")
        for key, unit, label, scale, suffix in (
            (
                "trajectory_agreement_score",
                "score_0_100",
                "Token prefix agreement",
                1,
                "/100",
            ),
            (
                "trajectory_full_match_rate",
                "fraction",
                "Exact token-sequence matches",
                100,
                "%",
            ),
        ):
            if key not in metrics:
                continue
            metric = metrics[key]
            expected = {
                "name": key,
                "unit": unit,
                "direction": "higher_is_better",
                "method": TrajectoryAgreementAnalysis.name,
                "version": TrajectoryAgreementAnalysis.version,
            }
            if metric["definition"] != expected:
                lines.append(
                    f"    {key}: unavailable for interpretation (unexpected metric identity)"
                )
            else:
                lines.append(f"    {label}: {metric['value'] * scale:.6g}{suffix}")
        diagnostics = analysis["diagnostics"].get("divergence")
        if diagnostics is not None:
            lines.append(
                f"    Divergent prompts: {diagnostics['diverged_prompts']}/{diagnostics['compared_prompts']}"
            )
            lines.append(
                f"    Trajectory evidence: {diagnostics['status']}; unavailable prompts: {diagnostics['unavailable_prompts']}"
            )
            if diagnostics["median_first_divergence"] is not None:
                lines.append(
                    f"    First divergence (zero-based): earliest {diagnostics['earliest_first_divergence']}, median {diagnostics['median_first_divergence']}"
                )
            lines.append(f"    Length mismatches: {diagnostics['length_mismatches']}")
            for category in diagnostics["by_category"]:
                lines.append(
                    f"      Category {html.escape(json.dumps(category['category'], ensure_ascii=True))}: {category['diverged_prompts']}/{category['n_prompts']} diverged"
                )
            for row in diagnostics["most_divergent"]:
                lines.append(
                    f"      {html.escape(json.dumps(row['id'], ensure_ascii=True))}: first divergence at token {row['first_divergence']}"
                )
        elif "per_prompt" in analysis["diagnostics"]:
            diverged = [
                row
                for row in analysis["diagnostics"]["per_prompt"]
                if not row.get("matched")
            ]
            lines.append(f"    Divergent prompts: {len(diverged)}")
            for row in diverged[:10]:
                lines.append(
                    f"      {row['id']}: first divergence at token {row['first_divergence']}"
                )
    for role in _ROLES:
        timing = manifest["systems"][role]
        if timing.get("available"):
            lines.append(
                f"  {role} mean process wall time: {timing['mean_seconds']:.6g}s (includes startup and model loading)"
            )
    performance = manifest.get("performance")
    if performance is not None:
        if performance["available"]:
            lines.append(
                f"  Cold-process request latency: {performance['absolute_delta']:+.6g}s ({performance['direction']})"
            )
            if performance["relative_delta"] is not None:
                lines.append(
                    f"    Relative change: {performance['relative_delta'] * 100:+.6g}%"
                )
            else:
                lines.append(
                    f"    Relative change unavailable: {performance['relative_unavailable_reason']}"
                )
            lines.append(f"    {performance['limitations']}")
        else:
            lines.append(f"  Performance impact unavailable: {performance['reason']}")
        lines.append(
            "  TTFT, decode throughput, inter-token latency, and device/KV memory: unavailable."
        )
    if not manifest["analyses"]:
        lines.append("  Behavioral impact unavailable: analysis did not complete.")
    if manifest.get("comparison_status") not in {None, "VALID"}:
        lines.append(
            "  No candidate benefit is inferred from an invalid or incomplete comparison."
        )
    lines.extend(("", "## Verdict:", f"  {manifest['verdict']}"))
    if "policy" in manifest:
        lines.extend(render_policy_evaluation(manifest["policy"]))
    else:
        lines.extend(
            (
                "  VERIFIED means comparison and analysis completed within the stated scope.",
                "  No task-quality or performance acceptance policy was evaluated.",
            )
        )
    lines.append("")
    return "\n".join(lines)
