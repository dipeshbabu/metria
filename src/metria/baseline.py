"""Immutable, content-addressed reference baselines with explicit freshness.

A baseline is a completed verification bundle captured byte for byte into a
local store, together with the exact recipe that produced it and a receipt
binding both to their hashes, the reference run's identity, the verifier
implementation, the hardware fingerprint and an expiry. The baseline ID is the
SHA-256 of the receipt's canonical content, and the directory is named by it,
so any later edit is detectable and nothing is silently replaced.

``check`` decides, per metric family, whether the saved reference is eligible
for a new recipe. Behavior and quality need an identical reference
configuration, workload, checks, capture environment, analyses and verifier
implementation, and an unexpired baseline. Timing and memory additionally need
an explicit performance reuse window chosen at capture and the same hardware;
by default they are never reused. Anything ineligible falls back to a fresh
paired run, and the reasons say why.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TextIO

from .recipes import (
    load_study_recipe,
    run_spec_to_data,
    study_recipe_digest,
    study_recipe_to_data,
)
from .records import load_run_record

SCHEMA = "metria.baseline.v1"
CHECK_SCHEMA = "metria.baseline_check.v1"
RECEIPT = "receipt.json"
RECIPE = "recipe.json"
BUNDLE_FILES = (
    "verification.json",
    "reference.run.json",
    "candidate.run.json",
    "report.md",
)
FAMILIES = ("behavior", "quality", "timing", "memory")
_REUSABLE_VERDICTS = frozenset({"VERIFIED", "PASS", "FAIL"})
_SPEC_REASONS = {
    "model": "model or tokenizer changed",
    "runtime": "runtime, deployment or trust configuration changed",
    "scenario": "generation settings changed",
    "measurements": "measurement methodology changed",
    "trial_policy": "trial policy changed",
    "treatments": "treatments changed",
}


class BaselineError(ValueError):
    """A baseline is malformed, tampered, incomplete or cannot be created."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def _parse_time(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise BaselineError(f"baseline {name} must be a timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise BaselineError(f"baseline {name} is not an ISO timestamp") from exc
    if parsed.tzinfo is None:
        raise BaselineError(f"baseline {name} must include a timezone")
    return parsed


def _now(now: datetime | None) -> datetime:
    return now if now is not None else datetime.now(timezone.utc)


def _recipe_controls(recipe: Any) -> dict[str, Any]:
    data = study_recipe_to_data(recipe)
    return {
        "reference_sha256": _sha256(_canonical(run_spec_to_data(recipe.study.runs[0]))),
        "measurement_configs_sha256": _sha256(_canonical(data["measurement_configs"])),
        "environment_sha256": _sha256(_canonical(data["environment"])),
        "analyses": sorted(recipe.study.comparison.analyses),
    }


def _read_bundle(evidence: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    from .pilot import _read_evidence

    try:
        manifest, hashes = _read_evidence(evidence.resolve())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise BaselineError(f"verification bundle is invalid: {exc}") from exc
    files = {name: (evidence / name).read_bytes() for name in BUNDLE_FILES}
    if {name: _sha256(data) for name, data in files.items()} != hashes:
        raise BaselineError("verification bundle changed while it was read")
    return manifest, files


def _check_reusable(manifest: Mapping[str, Any], allow_synthetic: bool) -> None:
    if manifest.get("verdict") not in _REUSABLE_VERDICTS:
        raise BaselineError(
            f"a {manifest.get('verdict')} verification cannot become a baseline"
        )
    reference = manifest["records"]["reference"]
    if reference.get("status") != "completed" or reference.get("evidence_gaps"):
        raise BaselineError("the reference run is incomplete or has evidence gaps")
    if manifest.get("fixture_only") and not allow_synthetic:
        raise BaselineError(
            "synthetic demo evidence cannot become a baseline without --allow-synthetic"
        )


def _receipt_core(
    manifest: Mapping[str, Any],
    files: Mapping[str, bytes],
    recipe_bytes: bytes,
    recipe: Any,
    *,
    created: datetime,
    max_age: timedelta,
    performance_max_age: timedelta | None,
) -> dict[str, Any]:
    reference = manifest["records"]["reference"]
    return {
        "schema": SCHEMA,
        "created_at": _timestamp(created),
        "expires_at": _timestamp(created + max_age),
        "performance_max_age_s": None
        if performance_max_age is None
        else int(performance_max_age.total_seconds()),
        "source": {
            "files_sha256": {
                name: _sha256(data) for name, data in sorted(files.items())
            },
            "recipe_sha256": _sha256(recipe_bytes),
            "recipe_digest": manifest["recipe_digest"],
            "scope": manifest["scope"],
            "verdict": manifest["verdict"],
            "fixture_only": bool(manifest.get("fixture_only", False)),
            "implementation": manifest.get("implementation"),
            "hardware_sha256": _sha256(_canonical(manifest.get("hardware"))),
        },
        "reference": {
            "run_id": reference["run_id"],
            "record_digest": reference["record_digest"],
            "evidence_digest": reference["evidence_digest"],
        },
        "controls": _recipe_controls(recipe),
    }


def capture(
    evidence: str | Path,
    recipe_path: str | Path,
    store: str | Path,
    *,
    max_age: timedelta,
    performance_max_age: timedelta | None = None,
    allow_synthetic: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Copy a completed bundle and its recipe into ``store`` under a new ID."""

    if max_age <= timedelta(0):
        raise BaselineError("baseline max age must be positive")
    if performance_max_age is not None and not (
        timedelta(0) < performance_max_age <= max_age
    ):
        raise BaselineError("performance max age must be positive and within max age")
    manifest, files = _read_bundle(Path(evidence))
    _check_reusable(manifest, allow_synthetic)
    recipe_bytes = Path(recipe_path).read_bytes()
    recipe = load_study_recipe(recipe_path)
    if study_recipe_digest(recipe) != manifest["recipe_digest"]:
        raise BaselineError("the recipe is not the one that produced this evidence")
    core = _receipt_core(
        manifest,
        files,
        recipe_bytes,
        recipe,
        created=_now(now),
        max_age=max_age,
        performance_max_age=performance_max_age,
    )
    baseline_id = _sha256(_canonical(core))
    receipt = {**core, "baseline_id": baseline_id}
    root = Path(store)
    root.mkdir(parents=True, exist_ok=True)
    for existing in _receipts(root):
        if existing["source"]["files_sha256"] == core["source"]["files_sha256"]:
            raise BaselineError(
                f"this evidence is already baseline {existing['baseline_id']}"
            )
    _publish(
        root, baseline_id, {**files, RECIPE: recipe_bytes, RECEIPT: _pretty(receipt)}
    )
    return receipt


def _pretty(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _publish(root: Path, baseline_id: str, files: Mapping[str, bytes]) -> None:
    staging = Path(tempfile.mkdtemp(prefix=".capture-", dir=root))
    try:
        for name, data in files.items():
            with (staging / name).open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        # rename never replaces an existing directory, so a concurrent capture
        # of the same baseline fails instead of overwriting retained evidence.
        os.rename(staging, root / baseline_id)
    except OSError as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise BaselineError(f"could not create baseline {baseline_id}: {exc}") from exc


def _receipts(root: Path) -> list[dict[str, Any]]:
    found = []
    for path in sorted(root.glob(f"*/{RECEIPT}")):
        if path.parent.name.startswith("."):
            continue
        try:
            found.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return found


def validate(path: str | Path) -> dict[str, Any]:
    """Re-verify every byte of a stored baseline; return its receipt."""

    directory = Path(path)
    if not directory.is_dir() or directory.is_symlink():
        raise BaselineError(f"{directory.name}: not a baseline directory")
    present = {item.name for item in directory.iterdir()}
    expected = {*BUNDLE_FILES, RECIPE, RECEIPT}
    if present != expected:
        missing, extra = sorted(expected - present), sorted(present - expected)
        raise BaselineError(
            f"baseline files differ: missing {missing}, unexpected {extra}"
        )
    try:
        receipt = json.loads((directory / RECEIPT).read_text(encoding="utf-8"))
    except ValueError as exc:
        raise BaselineError("baseline receipt is not valid JSON") from exc
    if not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA:
        raise BaselineError("not a metria.baseline.v1 receipt")
    core = {key: value for key, value in receipt.items() if key != "baseline_id"}
    baseline_id = _sha256(_canonical(core))
    if receipt.get("baseline_id") != baseline_id or directory.name != baseline_id:
        raise BaselineError("baseline receipt or directory name was altered")
    manifest, files = _read_bundle(directory)
    source = receipt["source"]
    if {name: _sha256(data) for name, data in files.items()} != source["files_sha256"]:
        raise BaselineError("baseline evidence bytes do not match the receipt")
    recipe_bytes = (directory / RECIPE).read_bytes()
    recipe = load_study_recipe(directory / RECIPE)
    if (
        _sha256(recipe_bytes) != source["recipe_sha256"]
        or study_recipe_digest(recipe) != manifest["recipe_digest"]
        or _recipe_controls(recipe) != receipt["controls"]
    ):
        raise BaselineError("baseline recipe does not match the receipt")
    record = load_run_record(directory / "reference.run.json")
    if record.run_id != receipt["reference"]["run_id"]:
        raise BaselineError("baseline reference record does not match the receipt")
    _parse_time(receipt["created_at"], "created_at")
    _parse_time(receipt["expires_at"], "expires_at")
    return receipt


def _current_implementation(stored: Any) -> dict[str, str]:
    from . import __version__

    current = {"name": "metria.verify", "version": __version__}
    if isinstance(stored, Mapping) and "source_sha256" in stored:
        from .verification_trials import _implementation_digest

        current["source_sha256"] = _implementation_digest()
    return current


def _identity_reasons(
    receipt: Mapping[str, Any], baseline_recipe: Any, recipe: Any
) -> list[str]:
    reasons = []
    stored = receipt["controls"]
    current = _recipe_controls(recipe)
    if current["reference_sha256"] != stored["reference_sha256"]:
        old = run_spec_to_data(baseline_recipe.study.runs[0])
        new = run_spec_to_data(recipe.study.runs[0])
        for key in sorted(set(old) | set(new)):
            if old.get(key) != new.get(key):
                reasons.append(_SPEC_REASONS.get(key, f"reference {key} changed"))
    if current["measurement_configs_sha256"] != stored["measurement_configs_sha256"]:
        reasons.append("workload or task checks changed")
    if current["environment_sha256"] != stored["environment_sha256"]:
        reasons.append("capture environment changed")
    if current["analyses"] != stored["analyses"]:
        reasons.append("analysis methodology changed")
    source = receipt["source"]
    if _current_implementation(source["implementation"]) != source["implementation"]:
        reasons.append("verifier implementation changed")
    if source["fixture_only"]:
        reasons.append("baseline is synthetic demo evidence")
    return reasons


def check(
    path: str | Path,
    recipe_path: str | Path,
    *,
    now: datetime | None = None,
    hardware: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Report per-metric eligibility of a stored baseline for ``recipe_path``."""

    receipt = validate(path)
    recipe = load_study_recipe(recipe_path)
    moment = _now(now)
    created = _parse_time(receipt["created_at"], "created_at")
    expires = _parse_time(receipt["expires_at"], "expires_at")
    age = moment - created
    shared = _identity_reasons(receipt, load_study_recipe(Path(path) / RECIPE), recipe)
    if moment >= expires:
        shared.append(f"baseline expired at {receipt['expires_at']}")
    if age < timedelta(0):
        shared.append("baseline was created in the future")
    performance = list(shared)
    window = receipt["performance_max_age_s"]
    if window is None:
        performance.append("no performance reuse window was set at capture")
    elif age > timedelta(seconds=window):
        performance.append("performance evidence is older than its reuse window")
    if hardware is None:
        from .hardware import capture_hardware_fingerprint

        hardware = capture_hardware_fingerprint().to_mapping()
    if _sha256(_canonical(hardware)) != receipt["source"]["hardware_sha256"]:
        performance.append("hardware differs from the baseline")
    families = {
        name: {"eligible": not reasons, "reasons": reasons}
        for name, reasons in (
            ("behavior", shared),
            ("quality", shared),
            ("timing", performance),
            ("memory", performance),
        )
    }
    reusable = [name for name in FAMILIES if families[name]["eligible"]]
    return {
        "schema": CHECK_SCHEMA,
        "baseline_id": receipt["baseline_id"],
        "recipe_digest": study_recipe_digest(recipe),
        "checked_at": _timestamp(moment),
        "age_s": int(age.total_seconds()),
        "families": families,
        "decision": "reuse_reference" if reusable else "fresh_paired_run",
        "reusable": reusable,
        "fresh_paired_run_required_for": [n for n in FAMILIES if n not in reusable],
    }


def add_baseline_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "baseline", help="capture, validate and check immutable reference baselines"
    )
    commands = parser.add_subparsers(dest="baseline_command", required=True)
    cap = commands.add_parser(
        "capture", help="store a completed verification as a baseline"
    )
    cap.add_argument(
        "--evidence", type=Path, required=True, help="verification output directory"
    )
    cap.add_argument(
        "--recipe", type=Path, required=True, help="recipe that produced it"
    )
    cap.add_argument(
        "--store", type=Path, required=True, help="local baseline store directory"
    )
    cap.add_argument("--max-age-days", type=float, required=True)
    cap.add_argument(
        "--performance-max-age-hours",
        type=float,
        help="allow timing and memory reuse on identical hardware within this window",
    )
    cap.add_argument("--allow-synthetic", action="store_true", help=argparse.SUPPRESS)
    for name, text in (
        ("inspect", "print a baseline receipt after validating it"),
        ("validate", "verify a stored baseline's integrity"),
    ):
        sub = commands.add_parser(name, help=text)
        sub.add_argument("baseline", type=Path)
    chk = commands.add_parser("check", help="decide per-metric reuse for a recipe")
    chk.add_argument("baseline", type=Path)
    chk.add_argument("--recipe", type=Path, required=True)
    chk.add_argument("--json", dest="json_output", action="store_true")


def baseline_command(args: Any, stdout: TextIO) -> int:
    if args.baseline_command == "capture":
        performance = (
            None
            if args.performance_max_age_hours is None
            else timedelta(hours=args.performance_max_age_hours)
        )
        receipt = capture(
            args.evidence,
            args.recipe,
            args.store,
            max_age=timedelta(days=args.max_age_days),
            performance_max_age=performance,
            allow_synthetic=args.allow_synthetic,
        )
        stdout.write(
            f"Captured baseline {receipt['baseline_id']}\n"
            f"Expires {receipt['expires_at']}\n"
        )
        return 0
    if args.baseline_command == "validate":
        receipt = validate(args.baseline)
        stdout.write(f"Baseline {receipt['baseline_id']} is intact\n")
        return 0
    if args.baseline_command == "inspect":
        stdout.write(
            json.dumps(validate(args.baseline), indent=2, sort_keys=True) + "\n"
        )
        return 0
    result = check(args.baseline, args.recipe)
    if args.json_output:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        stdout.write(f"Baseline {result['baseline_id']}: {result['decision']}\n")
        for name, family in result["families"].items():
            state = "reuse" if family["eligible"] else "fresh run"
            stdout.write(f"  {name}: {state}\n")
            for reason in family["reasons"]:
                stdout.write(f"    - {reason}\n")
    return 0 if result["decision"] == "reuse_reference" else 3
