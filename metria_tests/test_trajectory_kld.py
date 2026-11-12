"""Staged decode-time distribution capture and trajectory KL divergence."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import pytest

import metria.runtimes.llamacpp as llamacpp_module
from metria.fidelity.measurement_math import approximate_topk_kl
from metria.measurements import (
    TokenDistributionProtocol,
    TrajectoryDivergenceAnalysis,
    compare_trajectory_distributions,
)
from metria.models import RunRecord, RunSpec, RunStatus
from metria.protocols import (
    CaptureRequest,
    InferenceBatch,
    InferenceRequest,
    MeasurementResult,
)
from metria.runtimes.llamacpp_capture import read_token_capture, read_token_trajectory

PATCHES = Path(__file__).resolve().parents[1] / "tools" / "qualification"

# Two fixed next-token distributions over a 3-token visible support.
SHARP = [[1, math.log(0.7)], [2, math.log(0.2)], [3, math.log(0.05)]]
FLAT = [[1, math.log(0.4)], [2, math.log(0.35)], [3, math.log(0.2)]]


def _rows(*steps: dict[str, Any]) -> str:
    return "\n".join(json.dumps(step) for step in steps) + "\n"


# --- native provider ABI ---------------------------------------------------


def test_reader_returns_tokens_and_distributions(tmp_path: Path) -> None:
    path = tmp_path / "capture.jsonl"
    path.write_text(
        _rows(
            {"step": 0, "token_id": 1, "top_logprobs": SHARP},
            {"step": 1, "token_id": 2, "top_logprobs": FLAT},
        ),
        encoding="utf-8",
    )
    tokens, distributions = read_token_capture(path, top_k=3)
    assert tokens == [1, 2]
    assert distributions is not None
    assert [token for token, _ in distributions[0]] == [1, 2, 3]
    assert distributions[1][0][1] == pytest.approx(math.log(0.4))


def test_qualified_reader_still_rejects_distribution_fields(tmp_path: Path) -> None:
    path = tmp_path / "capture.jsonl"
    path.write_text(
        _rows({"step": 0, "token_id": 1, "top_logprobs": SHARP}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="fields"):
        read_token_trajectory(path)


@pytest.mark.parametrize(
    ("top_logprobs", "match"),
    [
        ([[1, -0.1], [1, -0.2]], "repeats"),
        ([[1, -2.0], [2, -0.1]], "sorted"),
        ([[1, 0.5]], "<= 0"),
        ([[1, -0.01], [2, -0.02]], "mass exceeds"),
        ([[-1, -0.1]], "non-negative"),
        ([[1, -0.1, 3]], "pairs"),
        ([], "between 1 and"),
        ([[1, -1]] * 4, "between 1 and"),
    ],
)
def test_reader_rejects_malformed_distributions(
    tmp_path: Path, top_logprobs: list[Any], match: str
) -> None:
    path = tmp_path / "capture.jsonl"
    path.write_text(
        _rows({"step": 0, "token_id": 1, "top_logprobs": top_logprobs}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=match):
        read_token_capture(path, top_k=3)


def test_reader_rejects_missing_nonfinite_and_changing_width(tmp_path: Path) -> None:
    path = tmp_path / "capture.jsonl"
    path.write_text(_rows({"step": 0, "token_id": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="staged top-k patch"):
        read_token_capture(path, top_k=3)

    path.write_text(
        '{"step":0,"token_id":1,"top_logprobs":[[1,-Infinity]]}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="non-finite"):
        read_token_capture(path, top_k=3)

    path.write_text(
        _rows(
            {"step": 0, "token_id": 1, "top_logprobs": SHARP},
            {"step": 1, "token_id": 1, "top_logprobs": SHARP[:2]},
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="width changed"):
        read_token_capture(path, top_k=3)

    with pytest.raises(ValueError, match="top_k"):
        read_token_capture(path, top_k=0)


def test_staged_patch_layers_on_the_unchanged_qualified_patch() -> None:
    staged = (PATCHES / "staged" / "llamacpp-topk-capture.patch").read_text(
        encoding="utf-8"
    )
    assert "KV_FIDELITY_TOPK" in staged
    assert "llama_get_logits_ith(ctx, -1)" in staged
    # The staged patch modifies lines the qualified patch introduced.
    assert '-                trajectory << "{\\"step\\":"' in staged
    qualified = (PATCHES / "llamacpp-capture.patch").read_text(encoding="utf-8")
    assert "KV_FIDELITY_TOPK" not in qualified


# --- llama.cpp runtime -----------------------------------------------------


def _llama_session(tmp_path: Path):
    from metria_tests.test_llamacpp_runtime import _session

    return _session(tmp_path)


def test_llamacpp_runtime_requests_and_returns_staged_distributions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, session = _llama_session(tmp_path)
    seen: dict[str, str] = {}

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen["topk"] = kwargs["env"]["KV_FIDELITY_TOPK"]
        Path(kwargs["env"]["KV_FIDELITY_TRAJECTORY"]).write_text(
            _rows(
                {"step": 0, "token_id": 1, "top_logprobs": SHARP},
                {"step": 1, "token_id": 3, "top_logprobs": FLAT},
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0, stdout="| out\n", stderr="")

    monkeypatch.setattr(llamacpp_module.subprocess, "run", fake_run)
    batch = session.infer(
        (InferenceRequest(prompt="p"),),
        capture=(
            CaptureRequest(kind="token_ids"),
            CaptureRequest(kind="token_top_logprobs"),
        ),
    )
    assert seen["topk"] == str(llamacpp_module.TOP_LOGPROBS_K)
    assert batch.captures["token_ids"] == ((1, 3),)
    assert len(batch.captures["token_top_logprobs"][0]) == 2
    invocation = adapter.observe(session)["invocations"][0]
    assert invocation["top_logprobs"]["qualification"] == "staged"


def test_llamacpp_token_only_capture_never_requests_distributions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, session = _llama_session(tmp_path)
    monkeypatch.setenv("KV_FIDELITY_TOPK", "5")
    seen: dict[str, Any] = {}

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen["topk"] = kwargs["env"].get("KV_FIDELITY_TOPK")
        Path(kwargs["env"]["KV_FIDELITY_TRAJECTORY"]).write_text(
            _rows({"step": 0, "token_id": 1}), encoding="utf-8"
        )
        return subprocess.CompletedProcess(argv, 0, stdout="| out\n", stderr="")

    monkeypatch.setattr(llamacpp_module.subprocess, "run", fake_run)
    batch = session.infer(
        (InferenceRequest(prompt="p"),), capture=(CaptureRequest(kind="token_ids"),)
    )
    assert seen["topk"] is None
    assert "token_top_logprobs" not in batch.captures


def test_llamacpp_probe_marks_distribution_capture_experimental(
    tmp_path: Path,
) -> None:
    from metria_tests.test_llamacpp_runtime import _files, _spec

    bin_dir, model = _files(tmp_path)
    report = llamacpp_module.LlamaCppAdapter().probe(_spec(bin_dir, model), {})
    marker = report.evidence["token_top_logprobs_capture"]
    assert marker["status"] == "experimental"


# --- comparison --------------------------------------------------------------


class _Session:
    def __init__(self, tokens, distributions):
        self.tokens, self.distributions = tokens, distributions

    def infer(self, requests, capture=()):
        assert {item.kind for item in capture} == {"token_ids", "token_top_logprobs"}
        return InferenceBatch(
            outputs=tuple("" for _ in requests),
            captures={
                "token_ids": self.tokens,
                "token_top_logprobs": self.distributions,
            },
        )


CONFIG = {"prompts": [{"id": "a", "prompt": "alpha"}, {"id": "b", "prompt": "beta"}]}


def _capture(tokens, distributions) -> MeasurementResult:
    return TokenDistributionProtocol().execute(
        _Session(tokens, distributions), {}, CONFIG
    )


def test_identical_runs_have_zero_divergence() -> None:
    capture = _capture(((1, 2), (3,)), ((SHARP, FLAT), (SHARP,)))
    assert capture.evidence["qualification"] == "staged"
    result = compare_trajectory_distributions(capture, capture)
    assert result.metrics["trajectory_kld_mean"].value == pytest.approx(0.0, abs=1e-12)
    assert result.evidence["scored_steps"] == 3
    assert result.evidence["qualification"] == "staged"


def test_scoring_stops_at_the_first_divergent_step() -> None:
    reference = _capture(((1, 2, 3), (3,)), ((SHARP, SHARP, SHARP), (SHARP,)))
    # Prompt a diverges at step 1, so step 2 has different context and is excluded.
    candidate = _capture(((1, 9, 3), (3,)), ((SHARP, FLAT, FLAT), (FLAT,)))
    result = compare_trajectory_distributions(reference, candidate)

    expected = approximate_topk_kl(
        {token: logprob for token, logprob in SHARP},
        {token: logprob for token, logprob in FLAT},
    )
    rows = {row["id"]: row for row in result.evidence["per_prompt"]}
    assert rows["a"]["first_divergence"] == 1
    assert rows["a"]["scored_steps"] == 2
    assert rows["a"]["max_kld"] == pytest.approx(expected)
    assert rows["b"]["scored_steps"] == 1
    assert result.metrics["trajectory_kld_mean"].value == pytest.approx(
        (0.0 + expected + expected) / 3
    )
    assert result.metrics["trajectory_kld_max"].value == pytest.approx(expected)
    assert result.metrics["trajectory_kld_mean"].coverage == 1.0


def test_prompt_without_shared_context_lowers_coverage() -> None:
    reference = _capture(((1,), ()), ((SHARP,), ()))
    candidate = _capture(((1,), (4,)), ((SHARP,), (FLAT,)))
    result = compare_trajectory_distributions(reference, candidate)
    assert result.metrics["trajectory_kld_mean"].coverage == 0.5
    assert result.evidence["notes"]


def test_comparison_rejects_unscorable_or_mismatched_evidence() -> None:
    empty = _capture(((), ()), ((), ()))
    other = _capture(((1,), (2,)), ((SHARP,), (SHARP,)))
    with pytest.raises(RuntimeError, match="no decode step"):
        compare_trajectory_distributions(empty, other)

    narrow = _capture(((1,), (2,)), ((SHARP[:2],), (SHARP[:2],)))
    with pytest.raises(ValueError, match="widths differ"):
        compare_trajectory_distributions(other, narrow)

    with pytest.raises(RuntimeError, match="every decode step"):
        _capture(((1, 2), (3,)), ((SHARP,), (SHARP,)))

    with pytest.raises(ValueError, match="not staged distribution capture"):
        compare_trajectory_distributions(MeasurementResult(evidence={}), other)


def test_analysis_reads_run_local_capture_evidence() -> None:
    capture = _capture(((1,), (2,)), ((SHARP,), (FLAT,)))

    def record(run_id: str) -> RunRecord:
        return RunRecord(
            study_name="s",
            run_id=run_id,
            requested=RunSpec(model={}, runtime={}, scenario={}, measurements=()),
            resolved={},
            observed={},
            status=RunStatus.COMPLETED,
            evidence={
                "measurements": {TokenDistributionProtocol.name: capture.evidence}
            },
        )

    result = TrajectoryDivergenceAnalysis().analyze(record("r"), record("c"))
    assert result.metrics["trajectory_kld_mean"].value == pytest.approx(0.0, abs=1e-12)
