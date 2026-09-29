"""Deterministic human-readable projections of canonical verification results."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from typing import Any

from .measurements import TrajectoryAgreementAnalysis
from .measurements.impact_schema import NAME as IMPACT_NAME
from .measurements.impact_schema import SCHEMA as IMPACT_SCHEMA
from .measurements.impact_schema import VERSION as IMPACT_VERSION
from .policies import render_policy_evaluation
from .verification_schema import VERIFICATION_ROLES as _ROLES
from .verification_schema import VERIFICATION_SCOPE, VLLM_VERIFICATION_SCOPE


def render_verification(manifest: Mapping[str, Any]) -> str:
    """Render a concise report without raw prompts or configuration values."""
    lines = [
        "# Metria Verification",
        "",
        f"**Verdict: {manifest['verdict']}**",
        "",
        f"Recipe: `{manifest['recipe_digest']}`",
        _scope_label(manifest),
        "",
        "## Change:",
        _change_label(manifest),
        "",
        "## Evidence:",
    ]
    for role in _ROLES:
        record = manifest["records"][role]
        lines.append(f"  {role}: {record['status']} ({record['path']})")
        facts = record["observed"]
        lines.append(_facts_label(facts, manifest.get("scope")))
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
        lines.extend(_render_impact_details(analysis))
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
    lines.extend(_render_performance(manifest))
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


def _render_performance(manifest: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    performance = manifest.get("performance")
    if performance is not None:
        if performance["available"]:
            lines.append(
                f"  {performance.get('label', 'Cold-process request latency')}: {performance['absolute_delta']:+.6g}s ({performance['direction']})"
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
        if "metrics" in performance:
            lines.extend(_serving_metrics(performance["metrics"]))
        else:
            lines.append(
                "  TTFT, decode throughput, inter-token latency, and device/KV memory: unavailable."
            )
    return lines


def _serving_metrics(metrics: Mapping[str, Any]) -> list[str]:
    from .measurements.serving_metrics import DEFINITIONS

    lines = []
    for name, definition in DEFINITIONS.items():
        row = metrics.get(name, {})
        if row.get("available") is not True:
            lines.append(f"    {name}: unavailable")
        elif (row.get("unit"), row.get("method"), row.get("version")) == (
            definition.unit,
            definition.method,
            definition.version,
        ):
            lines.append(
                f"    {name}: {row['reference']:.6g} -> {row['candidate']:.6g} {definition.unit}"
            )
        else:
            lines.append(f"    {name}: incompatible metric identity")
    lines.append(
        "    Memory: native worker PyTorch allocator peaks and allocated KV tensor storage; excludes other processes and non-PyTorch allocations."
    )
    lines.append(
        "    Decode rate: generated tokens after the first divided by summed per-request stream intervals; unavailable when token chunks coalesce."
    )
    return lines


def _scope_label(manifest: Mapping[str, Any]) -> str:
    from .verification_schema import (
        GGUF_QUANTIZATION_SCOPE,
        LLAMACPP_BUILD_SCOPE,
        VLLM_UPGRADE_SCOPE,
    )

    if manifest.get("scope") == "local_vllm_serving_concurrency.v1":
        return "Scope: local vLLM streaming serving API; no HTTP/network timing"
    if manifest.get("fixture_only") is True:
        return "Scope: synthetic fixture; no real runtime or model qualification"
    if manifest.get("scope") == VERIFICATION_SCOPE:
        return "Scope: local llama.cpp CPU thread comparison"
    if manifest.get("scope") == VLLM_VERIFICATION_SCOPE:
        return "Scope: local vLLM prefix-cache comparison"
    if manifest.get("scope") == LLAMACPP_BUILD_SCOPE:
        return "Scope: local llama.cpp CPU build comparison"
    if manifest.get("scope") == GGUF_QUANTIZATION_SCOPE:
        return "Scope: local llama.cpp GGUF weight-quantization comparison"
    if manifest.get("scope") == VLLM_UPGRADE_SCOPE:
        return "Scope: local vLLM CPU runtime-stack upgrade"
    return "Scope: unrecognized verification contract"


def _change_label(manifest: Mapping[str, Any]) -> str:
    from .verification_schema import (
        GGUF_QUANTIZATION_SCOPE,
        LLAMACPP_BUILD_SCOPE,
        VLLM_UPGRADE_SCOPE,
    )

    change = manifest["change"]
    if manifest.get("scope") == "local_vllm_serving_concurrency.v1":
        return f"  Client concurrency: {change['reference']} -> {change['candidate']}; fixed engine capacity: {change['engine_capacity']}"
    if manifest.get("scope") == VLLM_UPGRADE_SCOPE:
        return f"  vLLM runtime environment: {change['reference']} -> {change['candidate']}; controlled CPU IDs: {list(change['cpu_binding'])}"
    if manifest.get("scope") == GGUF_QUANTIZATION_SCOPE:
        return f"  Observed tensor storage: {dict(change['reference'])} -> {dict(change['candidate'])} (Q8_0 conversion; mixed storage retained)"
    if manifest.get("scope") == LLAMACPP_BUILD_SCOPE:
        return (
            f"  Capture provider SHA256: {change['reference']} -> {change['candidate']}"
        )
    if manifest.get("scope") == VLLM_VERIFICATION_SCOPE:
        return f"  Prefix caching: {change['reference']} -> {change['candidate']}"
    return (
        f"  CPU threads: {change['reference_threads']} -> {change['candidate_threads']}"
    )


def _facts_label(facts: Mapping[str, Any], scope: Any) -> str:
    from .verification_schema import VLLM_UPGRADE_SCOPE

    if scope in {VLLM_UPGRADE_SCOPE, "local_vllm_serving_concurrency.v1"}:
        return f"    Observed runtime: {facts['runtime_version']}; context: {facts['context']}"
    if scope == VLLM_VERIFICATION_SCOPE:
        return f"    Observed prefix caching: {facts['prefix_caching']}; context: {facts['context']}"
    return f"    Observed threads: {facts['threads']}; context: {facts['context']}"


def _render_impact_details(analysis: Mapping[str, Any]) -> list[str]:
    if analysis["name"] != IMPACT_NAME:
        return []
    data = analysis["diagnostics"]
    if (data.get("schema"), data.get("method"), data.get("method_version")) != (
        IMPACT_SCHEMA,
        IMPACT_NAME,
        IMPACT_VERSION,
    ):
        return [
            "    Decision evidence unavailable: methodology is missing or incompatible."
        ]
    lines = []
    quality = data.get("quality", {})
    if quality.get("status") == "available":
        lines.append(
            f"    Task checks: reference {quality['reference_passed']}/{quality['check_count_per_role']}, candidate {quality['candidate_passed']}/{quality['check_count_per_role']} passed."
        )
        lines.append(
            f"    Checked workload: {quality['configured_prompts']}/{quality['total_prompts']} unique prompts; results describe declared checks only."
        )
        for row in quality["failed_candidate_checks"][:10]:
            lines.append(
                f"      Failed {html.escape(json.dumps(row['check_id']))} on {html.escape(json.dumps(row['prompt_id']))} ({row['kind']}); answer content retained by digest."
            )
    else:
        lines.append(
            f"    Task quality unavailable: {quality.get('reason', 'task checks were not configured')}."
        )
    variability = data.get("reference_variability", {})
    if variability.get("status") == "available":
        lines.append(
            f"    Reference repeatability: {variability['different_pairs']}/{variability['compared_pairs']} repeat pairs differed across {variability['unique_prompts']} prompts."
        )
        lines.append(f"    {variability['limitation']}")
    else:
        lines.append(
            f"    Reference variability unavailable: {variability.get('reason', 'repeated evidence is missing')}."
        )
    lines.append(
        "    Token differences are behavioral drift; task checks determine the declared quality outcomes."
    )
    return lines
