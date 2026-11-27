"""Collector-backed trust rules for ``remote_endpoint_change.v1``.

When runs name a host evidence collector, its signed facts upgrade identity
from server-claimed to collector-observed. A configured collector is then a
requirement: unverifiable, unstable or contradicting evidence is an evidence
gap, so it can never become a policy PASS. Without collectors the profile
keeps its claim-only behavior and authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .models import RunRecord, RunSpec
from .runtimes.endpoint_collector import DECLARED_TO_FACT, validate_collector
from .runtimes.endpoint_identity import validate_declared

_PER_RUN = ("url_env", "deployment", "role")


def collectors(left: RunSpec, right: RunSpec) -> tuple[Any, Any]:
    """Validate that both runs use a collector, or neither, consistently."""

    configs = [
        validate_collector(run.runtime.get("collector")) for run in (left, right)
    ]
    if (configs[0] is None) != (configs[1] is None):
        raise ValueError("either both endpoint runs name a collector or neither does")
    reference, candidate = configs
    if reference is None or candidate is None:
        return None, None
    pair: list[dict[str, Any]] = [reference, candidate]
    for config, role in zip(pair, ("reference", "candidate"), strict=True):
        if config["role"] != role:
            raise ValueError(f"the {role} run's collector must declare role {role}")
    shared = [{k: v for k, v in c.items() if k not in _PER_RUN} for c in pair]
    if shared[0] != shared[1]:
        raise ValueError("both collectors need the same keys_env and trust policy")
    if pair[0]["url_env"] == pair[1]["url_env"]:
        raise ValueError("reference and candidate must name different collector URLs")
    return pair[0], pair[1]


def collector_vary(left: RunSpec, right: RunSpec) -> set[str]:
    """Paths that legitimately differ when collectors corroborate a change."""

    reference, candidate = collectors(left, right)
    if reference is None:
        return set()
    paths = set()
    for key in _PER_RUN:
        if reference[key] != candidate[key]:
            paths |= {f"runtime.collector.{key}", f"resolved.collector.{key}"}
    declared = [validate_declared(run.runtime.get("declared")) for run in (left, right)]
    changed = [
        key
        for key, fact in DECLARED_TO_FACT.items()
        if key in declared[0] and declared[0].get(key) != declared[1].get(key)
    ]
    for key in changed:
        paths.add(f"observed.collector.facts.{DECLARED_TO_FACT[key]}")
    if changed:
        # Which engine values came from vLLM defaults can change with them.
        paths.add("observed.collector.facts.engine.defaulted")
    return paths


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def collector_gaps(record: RunRecord) -> list[str]:
    """Gaps for a run whose recipe names a collector."""

    if record.requested.runtime.get("collector") is None:
        return []
    observed = _mapping(record.observed.get("collector"))
    if observed.get("verified") is not True:
        errors = observed.get("errors") or ["no collector evidence was retained"]
        return [f"collector evidence could not be verified: {errors[0]}"]
    if observed.get("stable") is not True:
        return ["collector-observed facts changed during the run"]
    facts = _mapping(observed.get("facts"))
    declared = validate_declared(record.requested.runtime.get("declared"))
    gaps = []
    for key, fact in DECLARED_TO_FACT.items():
        if key in declared and fact in facts and facts[fact] != declared[key]:
            gaps.append(f"collector-observed {fact} contradicts runtime.declared.{key}")
    return gaps


def authority(record: RunRecord) -> str:
    observed = _mapping(record.observed.get("collector"))
    if observed.get("verified") is True and observed.get("stable") is True:
        return "collector_observed"
    return "claimed"
