"""Record a pilot decision against retained, content-verified Metria evidence."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from .onboarding import _json, _read
from .records import run_evidence_digest, run_record_digest, run_record_from_data
from .reporting import render_verification
from .verification_schema import VERIFICATION_ROLES, VERIFICATION_SCHEMA

SCHEMA = "metria.pilot_record.v1"


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 65536:
        raise ValueError(f"pilot {label} requires bounded nonempty text")
    return value


def validate_notes(value: Any) -> dict[str, Any]:
    required = {
        "kind",
        "reporter",
        "decision",
        "rationale",
        "acceptance_criteria",
        "setup_seconds",
        "friction",
        "would_reuse",
        "permission_to_share",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) - required - {"feedback_source"}
        or not required <= set(value)
    ):
        raise ValueError(
            "pilot notes require the documented decision and feedback fields"
        )
    if value["kind"] not in {"maintainer_validation", "participant_feedback"}:
        raise ValueError(
            "pilot kind must distinguish maintainer validation from participant feedback"
        )
    if value["decision"] not in {"keep", "reject", "inconclusive"}:
        raise ValueError("pilot decision must be keep, reject or inconclusive")
    for key in ("reporter", "rationale", "acceptance_criteria"):
        _text(value[key], key)
    seconds = value["setup_seconds"]
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not 0 <= seconds <= sys.float_info.max
    ):
        raise ValueError("pilot setup_seconds must be finite and nonnegative")
    if value["would_reuse"] is not None and not isinstance(value["would_reuse"], bool):
        raise ValueError("pilot would_reuse must be a boolean or null")
    if not isinstance(value["permission_to_share"], bool):
        raise ValueError("pilot permission_to_share must be an explicit boolean")
    friction = value["friction"]
    if not isinstance(friction, list) or len(friction) > 50:
        raise ValueError("pilot friction must be an array of at most 50 observations")
    for item in friction:
        _text(item, "friction observation")
    if value["kind"] == "participant_feedback" or "feedback_source" in value:
        _text(value.get("feedback_source"), "feedback_source")
    return dict(value)


def _bundle_file(directory: Path, name: str) -> bytes:
    path = directory / name
    if path.is_symlink() or path.resolve().parent != directory:
        raise ValueError("pilot evidence must remain within its local bundle")
    with path.open("rb") as stream:
        data = stream.read(32 * 1024 * 1024 + 1)
    if len(data) > 32 * 1024 * 1024:
        raise ValueError("pilot evidence file exceeds the 32 MiB bound")
    return data


def _read_evidence(directory: Path) -> tuple[dict[str, Any], dict[str, str]]:
    manifest_bytes = _bundle_file(directory, "verification.json")
    manifest = _json(manifest_bytes.decode("utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != VERIFICATION_SCHEMA:
        raise ValueError("pilot evidence requires a saved Metria verification bundle")
    for field in ("recipe_digest", "scope", "verdict"):
        _text(manifest.get(field), "verification " + field)
    records = manifest.get("records")
    if not isinstance(records, Mapping) or set(records) != set(VERIFICATION_ROLES):
        raise ValueError("pilot evidence requires both retained run records")
    files = {"verification.json": manifest_bytes}
    for role in VERIFICATION_ROLES:
        row = records[role]
        name = f"{role}.run.json"
        if not isinstance(row, Mapping) or row.get("path") != name:
            raise ValueError(
                "pilot record paths must name the local reference/candidate files"
            )
        content = _bundle_file(directory, name)
        record = run_record_from_data(_json(content.decode("utf-8")))
        if (
            row.get("record_digest") != run_record_digest(record)
            or row.get("evidence_digest") != run_evidence_digest(record)
            or row.get("run_id") != record.run_id
            or row.get("status") != record.status.value
        ):
            raise ValueError(
                "pilot run record content does not match the saved verification"
            )
        files[name] = content
    report = _bundle_file(directory, "report.md")
    try:
        expected_report = render_verification(manifest)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("pilot verification report fields are malformed") from exc
    if report.decode("utf-8").replace("\r\n", "\n") != expected_report:
        raise ValueError("pilot report does not match the saved verification")
    files["report.md"] = report
    hashes = {
        name: hashlib.sha256(content).hexdigest() for name, content in files.items()
    }
    return manifest, hashes


def record_pilot(evidence: str | Path, notes: Mapping[str, Any]) -> dict[str, Any]:
    """Retain an operator's decision; do not infer participant feedback or consent."""
    validated = validate_notes(notes)
    manifest, hashes = _read_evidence(Path(evidence).resolve())
    return {
        "schema": SCHEMA,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "notes": validated,
        "evidence": {
            "files_sha256": hashes,
            "recipe_digest": manifest["recipe_digest"],
            "scope": manifest["scope"],
            "verification_verdict": manifest["verdict"],
            "comparison_status": manifest.get("comparison_status"),
            "policy": manifest.get("policy"),
            "fixture_only": manifest.get("fixture_only", False),
        },
        "interpretation": "Operator-supplied decision and feedback provenance; hashes bind retained evidence, not independent re-evaluation or proof of external adoption.",
    }


def add_pilot_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "pilot",
        help="record a workload decision and feedback against retained evidence",
    )
    commands = parser.add_subparsers(dest="pilot_command", required=True)
    record = commands.add_parser(
        "record", help="retain a pilot decision; no external upload"
    )
    for name in ("evidence", "notes", "output"):
        record.add_argument("--" + name, type=Path, required=True)


def pilot_command(args: Any, stdout: TextIO) -> int:
    from .verification import _write_atomic

    if args.output.exists():
        raise FileExistsError("pilot output already exists; choose a new file")
    notes = _json(_read(args.notes))
    receipt = record_pilot(args.evidence, notes)
    _write_atomic(args.output, json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    stdout.write(
        f"Saved pilot record: {args.output}\nDecision: {receipt['notes']['decision']}\n"
    )
    return 0
