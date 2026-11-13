#!/usr/bin/env python3
"""Functional checks for a staged top-k llama.cpp capture provider.

STAGED: passing these checks shows the layered provider behaves as designed. It
does not qualify the provider; qualification still requires retained native
evidence (see docs/guides/staged-capabilities.md).

Checks, in order:

1. Without ``KV_FIDELITY_TOPK`` the staged binary's token stream equals the
   qualified binary's, so the patch leaves the qualified ABI untouched.
2. With it, every decode step carries a top-k distribution accepted by
   Metria's strict reader, and the token stream is still unchanged.
3. At temperature 0 the sampled token is the captured top-1 at every step, so
   the log probabilities are the model's own pre-sampling distribution.
4. Repeating an identical run through Metria's llama.cpp adapter gives a
   trajectory KL divergence of exactly zero.
5. A real numeric change (flash attention on) gives finite, non-negative KL.

Example (Linux or WSL, from a checkout with ``src`` on ``PYTHONPATH``)::

    python3 tools/qualification/staged/check_topk_provider.py \\
        --qualified-bin ~/builds/qualified/build/bin \\
        --staged-bin ~/builds/staged-topk/build/bin \\
        --model ~/models/stories260K.gguf
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from metria.measurements import (
    TokenDistributionProtocol,
    compare_trajectory_distributions,
)
from metria.models import RunSpec, TreatmentSpec, TreatmentType
from metria.protocols import MeasurementResult
from metria.runtimes.llamacpp import TOP_LOGPROBS_K, LlamaCppAdapter
from metria.runtimes.llamacpp_capture import TopLogprobs, read_token_capture

_PROMPTS = (
    {"id": "story", "prompt": "Once upon a time"},
    {"id": "dog", "prompt": "The little dog"},
    {"id": "sun", "prompt": "One day the sun"},
)


class _Report:
    def __init__(self) -> None:
        self.failed = False

    def check(self, label: str, ok: bool, detail: str = "") -> None:
        suffix = f" - {detail}" if detail else ""
        print(f"{'PASS' if ok else 'FAIL'}  {label}{suffix}", flush=True)
        self.failed = self.failed or not ok


def _raw_capture(
    binary: Path, model: Path, *, tokens: int, top_k: int | None
) -> tuple[list[int], list[TopLogprobs] | None]:
    """Run llama-completion directly with only the capture variables set."""

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trajectory.jsonl"
        env = {**os.environ, "KV_FIDELITY_TRAJECTORY": str(path)}
        env.pop("KV_FIDELITY_TOPK", None)
        if top_k is not None:
            env["KV_FIDELITY_TOPK"] = str(top_k)
        subprocess.run(
            [
                str(binary),
                "-m",
                str(model),
                "-p",
                _PROMPTS[0]["prompt"],
                "-n",
                str(tokens),
                "-c",
                "256",
                "--seed",
                "1",
                "--temp",
                "0",
                "-no-cnv",
                "--no-display-prompt",
                "-t",
                "2",
            ],
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=True,
            timeout=300,
        )
        return read_token_capture(path, top_k=top_k)


def _check_provider_abi(
    report: _Report, args: argparse.Namespace, qualified: Path, staged: Path
) -> None:
    expected, _ = _raw_capture(qualified, args.model, tokens=args.tokens, top_k=None)
    plain, none = _raw_capture(staged, args.model, tokens=args.tokens, top_k=None)
    report.check(
        "token stream unchanged without KV_FIDELITY_TOPK",
        plain == expected and none is None,
        f"{len(expected)} tokens",
    )
    tokens, distributions = _raw_capture(
        staged, args.model, tokens=args.tokens, top_k=TOP_LOGPROBS_K
    )
    rows = distributions or []
    report.check(
        f"every step carries a validated top-{TOP_LOGPROBS_K} distribution",
        len(rows) == len(tokens) and all(len(row) == TOP_LOGPROBS_K for row in rows),
    )
    report.check("token stream unchanged with KV_FIDELITY_TOPK set", tokens == expected)
    report.check(
        "temperature-0 sampled token equals captured top-1 at every step",
        bool(rows)
        and len(rows) == len(tokens)
        and all(row[0][0] == token for row, token in zip(rows, tokens, strict=True)),
    )
    masses = [math.fsum(math.exp(logprob) for _, logprob in row) for row in rows]
    report.check(
        f"captured top-{TOP_LOGPROBS_K} mass is within (0, 1]",
        bool(masses) and all(0.0 < mass <= 1.0 + 1e-6 for mass in masses),
        f"min {min(masses):.4f}, max {max(masses):.4f}" if masses else "",
    )


def _measure(
    args: argparse.Namespace, staged: Path, *, flash_attention: bool
) -> MeasurementResult:
    spec = RunSpec(
        model={"path": str(args.model), "id": args.model.stem},
        runtime={
            "name": "llamacpp",
            "bin_dir": str(staged),
            "n_gpu_layers": 0,
            "flash_attention": flash_attention,
            "threads": 2,
        },
        scenario={
            "context": 256,
            "max_tokens": args.tokens,
            "temperature": 0.0,
            "seed": 1,
            "chat_template": False,
        },
        measurements=(TokenDistributionProtocol.name,),
        treatments=(
            TreatmentSpec(
                name="llamacpp.kv_cache",
                kind=TreatmentType.RUNTIME_FEATURE,
                config={"key_dtype": "f16", "value_dtype": "f16"},
            ),
        ),
    )
    adapter = LlamaCppAdapter()
    session = adapter.launch(adapter.resolve(spec, {}), {})
    try:
        return TokenDistributionProtocol().execute(
            session, {}, {"prompts": list(_PROMPTS)}
        )
    finally:
        session.close()


def _check_divergence(
    report: _Report, args: argparse.Namespace, staged: Path
) -> dict[str, Any]:
    reference = _measure(args, staged, flash_attention=False)
    repeat = compare_trajectory_distributions(
        reference, _measure(args, staged, flash_attention=False)
    )
    report.check(
        "self-comparison KL is exactly zero",
        repeat.metrics["trajectory_kld_mean"].value == 0.0,
        f"{repeat.evidence['scored_steps']} scored steps",
    )
    change = compare_trajectory_distributions(
        reference, _measure(args, staged, flash_attention=True)
    )
    mean = change.metrics["trajectory_kld_mean"].value
    report.check(
        "flash-attention change yields finite non-negative KL",
        math.isfinite(mean) and mean >= 0.0,
    )
    return {
        "change": "flash attention off -> on (f16 KV cache)",
        "trajectory_kld_mean_nats": mean,
        "trajectory_kld_max_nats": change.metrics["trajectory_kld_max"].value,
        "scored_steps": change.evidence["scored_steps"],
        "coverage": change.metrics["trajectory_kld_mean"].coverage,
        "first_divergence": {
            row["id"]: row["first_divergence"] for row in change.evidence["per_prompt"]
        },
        "qualification": change.evidence["qualification"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--qualified-bin", required=True, type=Path)
    parser.add_argument("--staged-bin", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--tokens", type=int, default=48)
    args = parser.parse_args(argv)
    args.model = args.model.expanduser().resolve()
    qualified = args.qualified_bin.expanduser().resolve()
    staged = args.staged_bin.expanduser().resolve()

    report = _Report()
    _check_provider_abi(
        report, args, qualified / "llama-completion", staged / "llama-completion"
    )
    summary = _check_divergence(report, args, staged)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
