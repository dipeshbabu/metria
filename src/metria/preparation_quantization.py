"""Pinned native GGUF conversion and installed quantization recipe preparation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, TextIO

from .measurements import TokenTrajectoryProtocol, TrajectoryAgreementAnalysis
from .measurements.verification_impact import VerificationImpactAnalysis
from .models import ComparisonPlan, RunSpec, StudySpec
from .policies import policy_from_data
from .preparation_cpu import qualify_cpu_providers
from .processes import run_process
from .recipes import StudyRecipe, study_recipe_digest, study_recipe_to_json
from .runtimes.gguf_identity import inspect_gguf
from .runtimes.llamacpp import _find_binary, _sha256_file
from .verification_quantization import (
    CAPTURE_KEY,
    CONVERSION_SCHEMA,
    QUANTIZATION_KEY,
    QUANTIZATION_VARIATIONS,
    _pin,
    build_route,
    validate_conversion,
)


def convert_gguf(
    source: Path,
    source_sha256: str,
    candidate: Path,
    quantizer: Path,
    receipt: Path,
    *,
    threads: int = 2,
    timeout: float = 300,
    quantizer_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Convert in a private staging directory, then publish without overwriting."""
    from .verification import _write_atomic

    if type(threads) is not int or threads <= 0:
        raise ValueError("quantization threads must be a positive integer")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or not 1 <= timeout <= 3600
    ):
        raise ValueError("quantization timeout must be between 1 and 3600 seconds")
    source = source.expanduser().resolve()
    candidate = candidate.expanduser().absolute()
    if candidate.resolve() == receipt.expanduser().resolve():
        raise ValueError("candidate model and conversion receipt need separate paths")
    if candidate.exists() or candidate.is_symlink() or receipt.exists():
        raise FileExistsError("choose new candidate and conversion receipt paths")
    expected = _pin(source_sha256)
    if _sha256_file(source) != expected:
        raise ValueError("source GGUF does not match its trusted SHA256")
    source_identity = inspect_gguf(source)
    if source_identity["tensor_types"].get("Q8_0"):
        raise ValueError(
            "requantizing an already quantized model is outside this scope"
        )
    quantizer = quantizer.expanduser().resolve()
    tool_pin = _sha256_file(quantizer)
    if quantizer_sha256 is not None and _pin(quantizer_sha256) != tool_pin:
        raise ValueError("quantizer does not match its trusted SHA256")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".metria-quantize-", dir=candidate.parent
    ) as directory:
        staging = Path(directory).resolve()
        if staging.parent != candidate.parent.resolve():
            raise ValueError("quantization staging directory escaped its destination")
        temporary = staging / "candidate.gguf"
        result = run_process(
            [str(quantizer), str(source), str(temporary), "Q8_0", str(threads)],
            timeout_s=timeout,
            max_output_bytes=2 * 1024 * 1024,
        )
        evidence: dict[str, Any] = {
            "schema": "metria.gguf_conversion_attempt.v1",
            "source_sha256": expected,
            "quantizer_sha256": tool_pin,
            "format": "Q8_0",
            "threads": threads,
            "returncode": result.returncode,
            "timed_out": result.timed_out,
            "stdout_truncated": result.stdout_truncated,
            "stderr_truncated": result.stderr_truncated,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        try:
            conversion = _complete_conversion(
                source,
                expected,
                source_identity,
                temporary,
                quantizer,
                tool_pin,
                result.returncode,
                result.timed_out,
            )
            # An exclusive hard link is atomic and avoids copying model weights.
            os.link(temporary, candidate)
        except (OSError, ValueError) as exc:
            evidence["status"] = "failed"
            evidence["error_type"] = type(exc).__name__
            evidence["message_sha256"] = hashlib.sha256(str(exc).encode()).hexdigest()
            _write_atomic(
                receipt, json.dumps(evidence, indent=2, allow_nan=False) + "\n"
            )
            raise
        evidence.update({"status": "converted", "conversion": conversion})
        _write_atomic(receipt, json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    return conversion


def _complete_conversion(
    source: Path,
    expected: str,
    source_identity: Mapping[str, Any],
    candidate: Path,
    quantizer: Path,
    tool_pin: str,
    returncode: int,
    timed_out: bool,
) -> Mapping[str, Any]:
    if returncode != 0 or timed_out or not candidate.is_file():
        raise ValueError(
            "native quantization failed; inspect the retained conversion receipt"
        )
    if _sha256_file(source) != expected or _sha256_file(quantizer) != tool_pin:
        raise ValueError("source or quantizer changed during conversion")
    conversion = {
        "schema": CONVERSION_SCHEMA,
        "format": "Q8_0",
        "quantizer_sha256": tool_pin,
        "source": {"sha256": expected, "gguf": dict(source_identity)},
        "candidate": {
            "sha256": _sha256_file(candidate),
            "gguf": inspect_gguf(candidate),
        },
    }
    return validate_conversion(conversion)


def prepare_gguf_quantization_recipe(
    bin_dir: Path,
    source: Path,
    candidate: Path,
    conversion: Mapping[str, Any],
    prompts: Sequence[Mapping[str, Any]],
    *,
    threads: int = 2,
    context: int = 256,
    max_tokens: int = 16,
    timeout: float = 300,
) -> StudyRecipe:
    directory = bin_dir.expanduser().resolve()
    provider = _find_binary(directory, "llama-completion")
    if provider is None:
        raise ValueError("bin-dir must contain the qualified native capture provider")
    validate_conversion(conversion)
    runs = tuple(
        RunSpec(
            model={
                "path": str(path.expanduser().resolve()),
                "sha256": conversion[role]["sha256"],
            },
            runtime={
                "name": "llamacpp",
                "bin_dir": str(directory),
                "threads": threads,
                "threads_batch": 1,
                "n_gpu_layers": 0,
                "flash_attention": False,
            },
            scenario={
                "context": context,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "seed": 42,
                "chat_template": False,
                "timeout": timeout,
            },
            measurements=(TokenTrajectoryProtocol.name,),
        )
        for role, path in (("source", source), ("candidate", candidate))
    )
    recipe = StudyRecipe(
        study=StudySpec(
            name="llamacpp-gguf-q8-weight-change",
            runs=runs,
            comparison=ComparisonPlan(
                vary=QUANTIZATION_VARIATIONS,
                control=frozenset(
                    {
                        "runtime",
                        "scenario",
                        "measurements",
                        "observed.gguf.tokenizer_sha256",
                        "observed.gguf.controls_sha256",
                        "observed.gguf.tensor_layout_sha256",
                    }
                ),
                analyses=(
                    TrajectoryAgreementAnalysis.name,
                    VerificationImpactAnalysis.name,
                ),
            ),
        ),
        measurement_configs={TokenTrajectoryProtocol.name: {"prompts": list(prompts)}},
        environment={
            CAPTURE_KEY: _sha256_file(provider),
            QUANTIZATION_KEY: dict(conversion),
        },
    )
    build_route(recipe)
    return recipe


def prepare_quantization_command(args: Any, stdout: TextIO) -> int:
    from .onboarding import _json, _read
    from .verification import _write_atomic

    output = args.output.expanduser().resolve()
    qualifications = [
        output.with_suffix(f".{role}.qualification.run.json")
        for role in ("reference", "candidate")
    ]
    receipt = output.with_suffix(".conversion.json")
    targets = (
        output,
        receipt,
        *qualifications,
        args.candidate_model.expanduser().resolve(),
    )
    if len(set(targets)) != len(targets):
        raise ValueError(
            "recipe, candidate model and evidence need distinct output paths"
        )
    if any(path.exists() for path in (output, receipt, *qualifications)):
        raise FileExistsError("choose new recipe, conversion and qualification paths")
    quantizer = args.quantizer or _find_binary(args.bin_dir, "llama-quantize")
    if quantizer is None:
        raise ValueError("supply --quantizer or a bin-dir containing llama-quantize")
    prompts = [
        _json(line) for line in _read(args.workload).splitlines() if line.strip()
    ]
    # Validate the checked workload before invoking a native transformation.
    from .measurements.checked_trajectory import CheckedTrajectoryProtocol

    CheckedTrajectoryProtocol().requirements({"prompts": prompts})
    policy = policy_from_data(_json(_read(args.policy))) if args.policy else None
    conversion = convert_gguf(
        args.model,
        args.model_sha256,
        args.candidate_model,
        quantizer,
        receipt,
        threads=args.threads,
        timeout=args.timeout,
        quantizer_sha256=args.quantizer_sha256,
    )
    recipe = prepare_gguf_quantization_recipe(
        args.bin_dir,
        args.model,
        args.candidate_model,
        conversion,
        prompts,
        threads=args.threads,
        context=args.context,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    recipe = replace(recipe, policy=policy)
    providers = {
        recipe.study.runs[0].runtime["bin_dir"]: recipe.environment[CAPTURE_KEY]
    }
    qualify_cpu_providers(recipe, qualifications, providers=providers)
    _write_atomic(output, study_recipe_to_json(recipe) + "\n")
    stdout.write(f"Prepared {output}\nRecipe: {study_recipe_digest(recipe)}\n")
    stdout.write(
        f"Observed candidate tensor storage: {dict(conversion['candidate']['gguf']['tensor_types'])}\n"
    )
    return 0
