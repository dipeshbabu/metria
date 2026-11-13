#!/usr/bin/env python3
"""STAGED: retain native trajectory-KLD runs and a corpus-KLD cross-check.

Runs Metria's real llama.cpp adapter with the staged top-k provider on a pinned
GGUF model whose attention heads support quantized KV caches. It records:

* a reference run (f16 KV) and an identical repeat, whose trajectory KL
  divergence must be exactly zero;
* candidate runs for each requested KV cache type, with trajectory KL
  divergence and token agreement against the reference;
* an independent corpus KL divergence for the same KV changes from
  ``llama-perplexity --kl-divergence`` over the pinned wikitext-2 test split,
  so the two estimators can be compared in magnitude and ranking.

Local paths are replaced with placeholders in every retained file. Passing
this does not promote a verification profile.

Example (Linux or WSL, ``src`` on ``PYTHONPATH``)::

    python3 tools/qualification/staged/qualify_trajectory_kld.py \\
        --staged-bin ~/builds/staged-topk/build/bin \\
        --model ~/models/qwen2.5-0.5b-instruct-q8_0.gguf \\
        --model-sha256 ca59ca7f... --corpus-cache ~/.cache/metria-corpora \\
        --output trajectory-kld-evidence
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from metria import RunSpec, TreatmentSpec, TreatmentType, execute_run
from metria.fidelity.corpora import ensure_wikitext_2
from metria.fidelity.runner import (
    _KLD_MEAN_RE,
    _PPL_RE,
    _RMS_DP_RE,
    _TOPP_RE,
    _first_float,
)
from metria.integrations.staged_captures import STAGED_CAPTURE_CHECK
from metria.measurements import (
    TokenDistributionProtocol,
    compare_trajectory_distributions,
    compare_trajectory_results,
)
from metria.protocols import MeasurementResult
from metria.records import run_record_from_data, run_record_to_json
from metria.runtimes.llamacpp import LlamaCppAdapter

PROMPTS = (
    {"id": "story", "prompt": "Once upon a time, in a small village by the sea,"},
    {"id": "science", "prompt": "Photosynthesis is the process by which plants"},
    {"id": "capital", "prompt": "The capital of France is"},
    {"id": "code", "prompt": "def fibonacci(n):\n    "},
    {"id": "history", "prompt": "The Industrial Revolution began in"},
    {"id": "list", "prompt": "Three common primary colors are"},
)


def _redact(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        for path, label in replacements.items():
            value = value.replace(path, label)
        return value
    if isinstance(value, dict):
        return {key: _redact(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, replacements) for item in value]
    return value


def _spec(args: argparse.Namespace, kv: str) -> RunSpec:
    return RunSpec(
        model={
            "path": str(args.model),
            "id": args.model_id,
            "sha256": args.model_sha256,
        },
        runtime={
            "name": "llamacpp",
            "bin_dir": str(args.staged_bin),
            "n_gpu_layers": 0,
            "flash_attention": True,
            "threads": args.threads,
        },
        scenario={
            "context": 512,
            "max_tokens": args.tokens,
            "temperature": 0.0,
            "seed": 1,
            "chat_template": False,
        },
        measurements=(TokenDistributionProtocol.name,),
        trial_policy={"capability_overrides": [STAGED_CAPTURE_CHECK]},
        treatments=(
            TreatmentSpec(
                name="llamacpp.kv_cache",
                kind=TreatmentType.RUNTIME_FEATURE,
                config={"key_dtype": kv, "value_dtype": kv},
            ),
        ),
    )


def _capture(
    args: argparse.Namespace, run_id: str, kv: str, replacements: dict[str, str]
) -> MeasurementResult:
    record = execute_run(
        study_name="staged-trajectory-kld",
        run_id=run_id,
        spec=_spec(args, kv),
        adapter=LlamaCppAdapter(),
        measurement=TokenDistributionProtocol(),
        measurement_config={"prompts": list(PROMPTS)},
        # Qualify the token-ID provider by its exact content identity.
        environment={"llama_cpp_token_ids_capture_sha256": args.provider_sha256},
    )
    if record.status.value != "completed":
        raise SystemExit(f"{run_id} did not complete: {record.status.value}")
    data = _redact(json.loads(run_record_to_json(record)), replacements)
    (args.output / f"{run_id}.run.json").write_text(
        run_record_to_json(run_record_from_data(data)) + "\n", encoding="utf-8"
    )
    return MeasurementResult(
        evidence=record.evidence["measurements"][TokenDistributionProtocol.name]
    )


def _trajectory_summary(
    reference: MeasurementResult, candidate: MeasurementResult
) -> dict[str, Any]:
    kld = compare_trajectory_distributions(reference, candidate)
    tokens = compare_trajectory_results(
        _as_trajectory(reference), _as_trajectory(candidate)
    )
    return {
        "trajectory_kld_mean_nats": kld.metrics["trajectory_kld_mean"].value,
        "trajectory_kld_max_nats": kld.metrics["trajectory_kld_max"].value,
        "kld_coverage": kld.metrics["trajectory_kld_mean"].coverage,
        "scored_steps": kld.evidence["scored_steps"],
        "trajectory_agreement_score": tokens.metrics[
            "trajectory_agreement_score"
        ].value,
        "first_divergence": {
            row["id"]: row["first_divergence"] for row in kld.evidence["per_prompt"]
        },
        "qualification": kld.evidence["qualification"],
    }


def _as_trajectory(result: MeasurementResult) -> MeasurementResult:
    evidence = dict(result.evidence)
    evidence.update(
        schema="metria.trajectory_capture.v1",
        method="kv_fidelity.decode_time_trajectory",
        method_version="0.3.4",
    )
    return MeasurementResult(evidence=evidence)


def _perplexity(
    binary: Path,
    args: argparse.Namespace,
    corpus: Path,
    kv: str,
    base: Path,
    *,
    build: bool,
) -> dict[str, Any]:
    command = [
        str(binary),
        "-m",
        str(args.model),
        "-f",
        str(corpus),
        "-c",
        "512",
        "--chunks",
        str(args.chunks),
        "-ngl",
        "0",
        "-t",
        str(args.threads),
        "-fa",
        "on",
        "-ctk",
        kv,
        "-ctv",
        kv,
    ]
    command += (
        ["--kl-divergence-base", str(base)]
        if build
        else ["--kl-divergence-base", str(base), "--kl-divergence"]
    )
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=7200,
        check=False,
    )
    text = completed.stdout + "\n" + completed.stderr
    if completed.returncode != 0:
        raise SystemExit(f"llama-perplexity failed for {kv}: {text[-800:]}")
    return {
        "kv": kv,
        "ppl": _first_float(_PPL_RE, text),
        "mean_kld_nats": None if build else _first_float(_KLD_MEAN_RE, text),
        "rms_dp_pct": None if build else _first_float(_RMS_DP_RE, text),
        "same_top_p_pct": None if build else _first_float(_TOPP_RE, text),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--staged-bin", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--corpus-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidates", nargs="+", default=["q8_0", "q4_0"])
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    args.model = args.model.expanduser().resolve()
    args.staged_bin = args.staged_bin.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    if hashlib.sha256(args.model.read_bytes()).hexdigest() != args.model_sha256:
        raise SystemExit("model SHA-256 does not match --model-sha256")
    args.provider_sha256 = hashlib.sha256(
        (args.staged_bin / "llama-completion").read_bytes()
    ).hexdigest()
    replacements = {
        str(args.model.parent): "${MODEL_DIR}",
        str(args.staged_bin.parents[1]): "${STAGED_BUILD}",
        str(Path(sys.prefix)): "${PYTHON_ENV}",
        str(Path.home()): "${HOME}",
    }

    reference = _capture(args, "reference-f16", "f16", replacements)
    summary: dict[str, Any] = {
        "schema": "metria.staged_trajectory_kld_qualification.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "id": args.model_id,
            "file": args.model.name,
            "sha256": args.model_sha256,
        },
        "provider_sha256": args.provider_sha256,
        "host": {
            "machine": platform.machine(),
            "system": platform.system(),
            "release": platform.release(),
        },
        "scenario": {
            "tokens": args.tokens,
            "prompts": len(PROMPTS),
            "flash_attention": True,
            "temperature": 0.0,
        },
        "self_comparison": _trajectory_summary(
            reference, _capture(args, "repeat-f16", "f16", replacements)
        ),
        "candidates": {},
        "corpus_kld": {},
    }
    for kv in args.candidates:
        summary["candidates"][kv] = _trajectory_summary(
            reference, _capture(args, f"candidate-{kv}", kv, replacements)
        )
        print(
            json.dumps({kv: summary["candidates"][kv]["trajectory_kld_mean_nats"]}),
            flush=True,
        )

    corpus = ensure_wikitext_2(args.corpus_cache) / "wiki.test.raw"
    perplexity = args.staged_bin / "llama-perplexity"
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "f16.kld"
        summary["corpus_kld"]["reference"] = _perplexity(
            perplexity, args, corpus, "f16", base, build=True
        )
        for kv in args.candidates:
            summary["corpus_kld"][kv] = _perplexity(
                perplexity, args, corpus, kv, base, build=False
            )
    summary["corpus"] = {
        "name": "wikitext-2-raw/wiki.test.raw",
        "sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "chunks": args.chunks,
        "context": 512,
    }
    summary = _redact(summary, replacements)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["self_comparison"]["trajectory_kld_mean_nats"] == 0.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
