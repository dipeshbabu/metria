"""Internal, explicit routes for qualified verification profiles."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .models import RunRecord
from .protocols import MeasurementProtocol, PairwiseAnalysis, RuntimeAdapter


@dataclass(frozen=True)
class VerificationRoute:
    scope: str
    adapters: Mapping[str, RuntimeAdapter]
    measurements: Mapping[str, MeasurementProtocol]
    analyses: Mapping[str, PairwiseAnalysis]
    evidence_gaps: Callable[[RunRecord], tuple[str, ...]]
    observed_facts: Callable[[RunRecord], dict[str, Any]]
    change: Mapping[str, Any]
    performance: Callable[[RunRecord, RunRecord, bool], dict[str, Any]]
    isolated: bool = False
    run_executor: Callable[..., RunRecord] | None = None
