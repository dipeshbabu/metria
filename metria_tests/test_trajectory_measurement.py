from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from metria.measurements import TokenTrajectoryProtocol, compare_trajectory_results
from metria.protocols import (
    CaptureRequest,
    InferenceBatch,
    InferenceRequest,
    MeasurementResult,
)


class FakeTrajectorySession:
    def __init__(self, token_batches: Sequence[Sequence[int]]) -> None:
        self._token_batches = tuple(tuple(tokens) for tokens in token_batches)
        self.requests: tuple[InferenceRequest, ...] = ()
        self.capture: tuple[CaptureRequest, ...] = ()
        self.closed = False

    def infer(
        self,
        requests: Sequence[InferenceRequest],
        capture: Sequence[CaptureRequest] = (),
    ) -> InferenceBatch:
        self.requests = tuple(requests)
        self.capture = tuple(capture)
        return InferenceBatch(
            outputs=tuple("unused" for _ in requests),
            captures={"token_ids": self._token_batches},
        )

    def reset(self, scope: str = "measurement") -> None:
        del scope

    def close(self) -> None:
        self.closed = True


def _config() -> dict[str, Any]:
    return {
        "prompts": (
            {"id": "p1", "category": "fact", "prompt": "private prompt one"},
            {"id": "p2", "category": "reasoning", "prompt": "private prompt two"},
        ),
        "generation": {"temperature": 0.0},
    }


def _capture_result(
    rows: Sequence[tuple[str, str, Sequence[int]]],
) -> MeasurementResult:
    return MeasurementResult(
        evidence={
            "schema": "metria.trajectory_capture.v1",
            "method": "kv_fidelity.decode_time_trajectory",
            "method_version": "0.3.4",
            "n_prompts": len(rows),
            "prompts": tuple(
                {
                    "id": prompt_id,
                    "prompt_sha256": prompt_hash,
                    "token_ids": tuple(tokens),
                    "token_count": len(tokens),
                    "category": None,
                }
                for prompt_id, prompt_hash, tokens in rows
            ),
        }
    )


def test_trajectory_protocol_captures_tokens_without_prompt_text() -> None:
    session = FakeTrajectorySession(((1, 2, 3), (4,)))
    protocol = TokenTrajectoryProtocol()

    result = protocol.execute(session, {}, _config())

    assert session.capture == (CaptureRequest(kind="token_ids"),)
    assert tuple(request.prompt for request in session.requests) == (
        "private prompt one",
        "private prompt two",
    )
    assert result.metrics["trajectory_mean_steps"].value == 2.0
    assert result.metrics["trajectory_nonempty_capture_rate"].value == 1.0
    assert result.evidence["prompts"][0]["token_ids"] == (1, 2, 3)
    serialized_evidence = repr(result.evidence)
    assert "private prompt one" not in serialized_evidence
    assert "private prompt two" not in serialized_evidence
    assert len(result.evidence["prompts"][0]["prompt_sha256"]) == 64


def test_trajectory_protocol_reports_partial_nonempty_capture() -> None:
    session = FakeTrajectorySession(((1, 2), ()))

    result = TokenTrajectoryProtocol().execute(session, {}, _config())

    assert result.metrics["trajectory_mean_steps"].value == 1.0
    assert result.metrics["trajectory_nonempty_capture_rate"].value == 0.5


def test_trajectory_protocol_validates_prompt_and_capture_shapes() -> None:
    protocol = TokenTrajectoryProtocol()
    with pytest.raises(ValueError, match="duplicate trajectory prompt id"):
        protocol.requirements(
            {
                "prompts": (
                    {"id": "same", "prompt": "a"},
                    {"id": "same", "prompt": "b"},
                )
            }
        )

    bad_session = FakeTrajectorySession(((1, 2),))
    with pytest.raises(RuntimeError, match="capture count does not match"):
        protocol.execute(bad_session, {}, _config())


@pytest.mark.parametrize(
    ("reference_tokens", "candidate_tokens", "expected_score"),
    [
        (((1, 2, 3, 4), (5, 6)), ((1, 2, 9, 4), (5,)), 50.0),
        (((1, 2),), ((1, 2),), 100.0),
        (((1, 2),), ((),), 0.0),
    ],
)
def test_pairwise_score_matches_kv_fidelity_v034_formula(
    reference_tokens: tuple[tuple[int, ...], ...],
    candidate_tokens: tuple[tuple[int, ...], ...],
    expected_score: float,
) -> None:
    hashes = tuple(f"{index + 1:064x}" for index in range(len(reference_tokens)))
    reference = _capture_result(
        tuple(
            (f"p{index}", hashes[index], tokens)
            for index, tokens in enumerate(reference_tokens)
        )
    )
    candidate = _capture_result(
        tuple(
            (f"p{index}", hashes[index], tokens)
            for index, tokens in enumerate(candidate_tokens)
        )
    )

    result = compare_trajectory_results(reference, candidate)

    assert result.metrics["trajectory_agreement_score"].value == expected_score


def test_pairwise_comparison_keeps_diagnostics_separate_from_run_metrics() -> None:
    reference = _capture_result(
        (
            ("p1", "1" * 64, (1, 2, 3)),
            ("p2", "2" * 64, (4, 5)),
        )
    )
    candidate = _capture_result(
        (
            ("p1", "1" * 64, (1, 2, 3)),
            ("p2", "2" * 64, (4, 9)),
        )
    )

    result = compare_trajectory_results(reference, candidate)

    assert result.metrics["trajectory_agreement_score"].value == 80.0
    assert result.metrics["trajectory_full_match_rate"].value == 0.5
    assert result.metrics["trajectory_mean_prefix_steps"].value == 2.0
    assert result.evidence["median_first_divergence"] == 1
    assert result.evidence["per_prompt"][0]["matched"] is True
    assert result.evidence["per_prompt"][1]["first_divergence"] == 1


def test_pairwise_comparison_rejects_prompt_identity_mismatch() -> None:
    reference = _capture_result((("p1", "1" * 64, (1, 2)),))
    candidate = _capture_result((("p1", "2" * 64, (1, 2)),))

    with pytest.raises(ValueError, match="prompt fingerprint differs"):
        compare_trajectory_results(reference, candidate)


def test_pairwise_comparison_rejects_different_prompt_sets() -> None:
    reference = _capture_result((("p1", "1" * 64, (1, 2)),))
    candidate = _capture_result((("p2", "2" * 64, (1, 2)),))

    with pytest.raises(ValueError, match="prompt sets differ"):
        compare_trajectory_results(reference, candidate)


def test_pairwise_comparison_rejects_two_empty_trajectories() -> None:
    reference = _capture_result((("p1", "1" * 64, ()),))
    candidate = _capture_result((("p1", "1" * 64, ()),))

    with pytest.raises(RuntimeError, match="both trajectories are empty"):
        compare_trajectory_results(reference, candidate)


def test_measurement_result_deeply_freezes_evidence() -> None:
    source: dict[str, Any] = {"nested": {"values": [1, 2]}}

    result = MeasurementResult(evidence=source)
    source["nested"]["values"].append(3)

    assert result.evidence["nested"]["values"] == (1, 2)
    with pytest.raises(TypeError):
        result.evidence["nested"]["new"] = "blocked"
