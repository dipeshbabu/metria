"""Repeat the qualified verifier with explicit warmup, isolation, and baseline identity."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .recipes import StudyRecipe, _json_value, study_recipe_digest
from .verification import VERIFICATION_SCOPE, _write_atomic, verify_recipe
from .verification_schema import (
    GGUF_QUANTIZATION_SCOPE,
    LLAMACPP_BUILD_SCOPE,
    VLLM_VERIFICATION_SCOPE,
)

TRIALS_SCHEMA = "metria.verification_trials.v1"


@dataclass(frozen=True)
class VerificationTrialPolicy:
    warmup_pairs: int = 0
    measured_pairs: int = 1

    def __post_init__(self) -> None:
        for name, minimum in (("warmup_pairs", 0), ("measured_pairs", 1)):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not minimum <= value <= 100
            ):
                raise ValueError(f"{name} must be an integer between {minimum} and 100")

    def to_data(self) -> dict[str, Any]:
        return {
            "warmup_pairs": self.warmup_pairs,
            "measured_pairs": self.measured_pairs,
            "isolation": "fresh verifier/runtime sessions for each pair; fresh llama.cpp process per prompt",
            "warmup_scope": "whole invocations, including filesystem/startup caches; no persistent loaded-engine warmup",
        }


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _json_value(value, path="trial_identity"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _implementation_digest() -> str:
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _baseline_identity(
    result: Mapping[str, Any], policy: VerificationTrialPolicy
) -> dict[str, Any] | None:
    if result.get("scope") not in {
        VERIFICATION_SCOPE,
        VLLM_VERIFICATION_SCOPE,
        LLAMACPP_BUILD_SCOPE,
        GGUF_QUANTIZATION_SCOPE,
    } or result.get("fixture_only"):
        return None
    records = result.get("records", {})
    pins = {}
    for role in ("reference", "candidate"):
        observed = records.get(role, {}).get("observed", {})
        identity = {
            key: observed.get(key) for key in ("model_sha256", "provider_sha256")
        }
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
            for value in identity.values()
        ):
            return None
        if result.get("scope") == VLLM_VERIFICATION_SCOPE:
            hardware = observed.get("runtime_hardware")
            if (
                not isinstance(hardware, Mapping)
                or hardware.get("status") != "observed"
            ):
                return None
            identity["runtime_hardware"] = hardware
        pins[role] = identity
    performance = result.get("performance", {})
    return {
        "recipe_digest": result["recipe_digest"],
        "implementation_source_sha256": _implementation_digest(),
        "observed_artifacts": pins,
        "hardware_digest": _digest(result["hardware"]),
        "analysis_methods": [
            (entry["name"], entry["version"]) for entry in result.get("analyses", ())
        ],
        "performance_method": performance.get("metric"),
        "performance_methodology": performance.get("methodology"),
        "trial_policy": policy.to_data(),
    }


def _collect_pairs(
    recipe: StudyRecipe, output: Path, policy: VerificationTrialPolicy
) -> tuple[
    list[dict[str, Any]],
    list[Mapping[str, Any]],
    dict[str, Any] | None,
    int,
    str,
    dict[str, Any] | None,
]:
    rows: list[dict[str, Any]] = []
    measurements: list[Mapping[str, Any]] = []
    identity: dict[str, Any] | None = None
    exit_code, status = 0, "completed"
    failure: dict[str, Any] | None = None
    try:
        for index in range(policy.warmup_pairs + policy.measured_pairs):
            warmup = index < policy.warmup_pairs
            name = f"{'warmup' if warmup else 'trial'}-{index:04d}"
            result = verify_recipe(recipe, output / name)
            data = result.to_data()
            rows.append(
                {
                    "path": name + "/verification.json",
                    "warmup": warmup,
                    "verdict": data["verdict"],
                    "exit_code": result.exit_code,
                    "record_digests": {
                        role: record["record_digest"]
                        for role, record in data["records"].items()
                    },
                }
            )
            if data["verdict"] not in {"VERIFIED", "PASS", "FAIL"}:
                exit_code = result.exit_code
                status = "interrupted" if exit_code == 130 else "verification_failed"
                break
            current = _baseline_identity(data, policy)
            if current is None or identity is not None and identity != current:
                exit_code, status = 3, "not_comparable"
                break
            identity = current
            if not warmup:
                measurements.append(data)
                if data["verdict"] == "FAIL":
                    exit_code, status = 1, "policy_failed"
    except KeyboardInterrupt:
        exit_code, status = 130, "interrupted"
    except (OSError, TypeError, ValueError) as exc:
        exit_code, status = 5, "execution_failed"
        failure = {
            "error_type": type(exc).__name__,
            "message_sha256": hashlib.sha256(str(exc).encode()).hexdigest(),
        }
    return rows, measurements, identity, exit_code, status, failure


def _latency_summary(
    measurements: list[Mapping[str, Any]], policy: VerificationTrialPolicy, status: str
) -> dict[str, Any]:
    latency: dict[str, Any] = {
        "available": False,
        "reason": "measured pairs are incomplete or lack compatible native timing",
    }
    if (
        status in {"completed", "policy_failed"}
        and len(measurements) == policy.measured_pairs
        and all(data.get("performance", {}).get("available") for data in measurements)
    ):
        values = {
            role: [data["performance"][role] for data in measurements]
            for role in ("reference", "candidate")
        }
        if all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
            for samples in values.values()
            for value in samples
        ):
            latency = {
                "available": True,
                "method": "metria.verification_trial_mean.v1",
                "unit": "seconds",
                "aggregation": "arithmetic mean across complete measured pairs",
                "sample_count": len(measurements),
                **{
                    role: {
                        "mean": sum(value / len(samples) for value in samples),
                        "samples": samples,
                    }
                    for role, samples in values.items()
                },
                "source_metric": measurements[0]["performance"].get("metric"),
                "source_methodology": measurements[0]["performance"].get("methodology"),
                "limitation": measurements[0]["performance"].get(
                    "limitations",
                    "Interpret latency using each pair's retained measurement boundary; no statistical speedup claim.",
                ),
            }
    return latency


def _persist_summary(output: Path, summary: Mapping[str, Any], measured: int) -> None:
    status = summary["status"]
    policy = summary["trial_policy"]
    report = [
        "# Verification trials",
        "",
        f"Status: {status}",
        f"Measured pairs retained: {measured}/{policy['measured_pairs']}",
        f"Warmup pairs requested: {policy['warmup_pairs']}",
        "",
        "Each pair retains the canonical verifier report and both run records.",
        "Acceptance is the recipe policy; no hardware/model-specific default threshold was applied.",
        "Baselines are eligible only for matching artifact, recipe/workload, runtime, hardware, method, and trial-policy identity.",
        "",
    ]
    _write_atomic(output / "report.md", "\n".join(report))
    _write_atomic(
        output / "verification-trials.json",
        json.dumps(
            _json_value(summary, path="trials"),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )


def execute_verification_trials(
    recipe: StudyRecipe,
    output_dir: str | Path,
    *,
    policy: VerificationTrialPolicy | None = None,
    baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Retain every pair, stop on invalid evidence, and summarize measured pairs.

    This is an explicit research utility around the same verifier lifecycle.
    It launches no background inference service and applies no guessed baseline
    or hardware-specific threshold. Acceptance remains the recipe's own policy.
    """
    if policy is None:
        policy = VerificationTrialPolicy()
    if not isinstance(policy, VerificationTrialPolicy):
        raise TypeError("policy must be a VerificationTrialPolicy")
    if not isinstance(recipe, StudyRecipe):
        raise TypeError("recipe must be a StudyRecipe")
    if baseline is not None and (
        not isinstance(baseline, Mapping) or baseline.get("schema") != TRIALS_SCHEMA
    ):
        raise ValueError("baseline must use the verification-trials schema")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    rows, measurements, identity, exit_code, status, failure = _collect_pairs(
        recipe, output, policy
    )
    baseline_key = _digest(identity) if identity is not None else None
    baseline_match = None
    if baseline is not None:
        baseline_match = (
            baseline.get("schema") == TRIALS_SCHEMA
            and baseline.get("status") == "completed"
            and baseline_key is not None
            and baseline.get("baseline_key") == baseline_key
        )
        if not baseline_match and exit_code in {0, 1}:
            exit_code, status = 3, "baseline_not_comparable"
    summary = {
        "schema": TRIALS_SCHEMA,
        "recipe_digest": study_recipe_digest(recipe),
        "trial_policy": policy.to_data(),
        "status": status,
        "exit_code": exit_code,
        "pairs": rows,
        "baseline_key": baseline_key,
        "baseline_identity": identity,
        "baseline_matches": baseline_match,
        "latency": _latency_summary(measurements, policy, status),
        "error": failure,
    }
    _persist_summary(output, summary, len(measurements))
    return summary
