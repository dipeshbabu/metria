"""Pairwise analyzer that derives trajectory fidelity from RunRecord evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..models import RunRecord
from ..protocols import MeasurementResult
from .trajectory import TokenTrajectoryProtocol, compare_trajectory_results


class TrajectoryAgreementAnalysis:
    """Derive KV Fidelity trajectory agreement from two run-local captures."""

    name = "kv_fidelity.trajectory_match"
    version = "0.3.4"

    def analyze(self, left: RunRecord, right: RunRecord) -> MeasurementResult:
        """Treat the left run as reference and the right run as candidate."""

        reference = self._capture_result(left)
        candidate = self._capture_result(right)
        return compare_trajectory_results(reference, candidate)

    @staticmethod
    def _capture_result(record: RunRecord) -> MeasurementResult:
        """Recover one trajectory capture from a run's measurement evidence."""

        measurements = record.evidence.get("measurements")
        if not isinstance(measurements, Mapping):
            raise ValueError(
                f"run {record.run_id!r} does not retain measurement evidence"
            )
        raw_evidence: Any = measurements.get(TokenTrajectoryProtocol.name)
        if not isinstance(raw_evidence, Mapping):
            raise ValueError(
                f"run {record.run_id!r} does not contain trajectory capture evidence"
            )
        return MeasurementResult(evidence=raw_evidence)
