import hashlib
import json
from io import StringIO

import pytest
from metria_tests.test_vllm_verify import execute
from metria_tests.test_vllm_verify import prefix_case as _prefix_fixture

from metria.cli import main
from metria.pilot import record_pilot, validate_notes

prefix_case = _prefix_fixture


def notes():
    return {
        "kind": "maintainer_validation",
        "reporter": "automated maintainer fixture",
        "decision": "inconclusive",
        "rationale": "Fixture validates evidence handling only.",
        "acceptance_criteria": "Retain the saved comparison and checks.",
        "setup_seconds": 12.5,
        "friction": ["Fixture setup only."],
        "would_reuse": None,
        "permission_to_share": False,
    }


@pytest.fixture
def bundle(prefix_case, tmp_path):
    return execute(prefix_case, tmp_path / "bundle").output_dir


def test_pilot_decision_binds_original_records_and_report_without_inventing_feedback(
    bundle,
):
    record = record_pilot(bundle, notes())
    assert record["schema"] == "metria.pilot_record.v1"
    assert record["notes"]["kind"] == "maintainer_validation"
    assert record["notes"]["would_reuse"] is None
    assert record["notes"]["permission_to_share"] is False
    assert record["evidence"]["verification_verdict"] == "VERIFIED"
    for name, digest in record["evidence"]["files_sha256"].items():
        assert hashlib.sha256((bundle / name).read_bytes()).hexdigest() == digest
    assert "private" not in json.dumps(record)


@pytest.mark.parametrize(
    "field,value",
    [
        ("kind", "satisfied-user"),
        ("decision", "pass"),
        ("reporter", ""),
        ("rationale", 0),
        ("setup_seconds", -1),
        ("setup_seconds", float("inf")),
        ("setup_seconds", True),
        ("would_reuse", "yes"),
        ("permission_to_share", None),
        ("friction", "none"),
        ("friction", [""]),
        ("friction", ["many"] * 51),
    ],
)
def test_invalid_or_implicit_pilot_feedback_is_rejected(field, value):
    with pytest.raises(ValueError):
        validate_notes({**notes(), field: value})


def test_participant_feedback_requires_supplied_source_and_explicit_permission():
    participant = {**notes(), "kind": "participant_feedback"}
    with pytest.raises(ValueError, match="feedback_source"):
        validate_notes(participant)
    participant["feedback_source"] = (
        "Participant-provided response retained locally; fixture only"
    )
    assert validate_notes(participant)["permission_to_share"] is False
    with pytest.raises(ValueError):
        validate_notes({"decision": "keep"})
    with pytest.raises(ValueError):
        validate_notes({**notes(), "unknown": "field"})


@pytest.mark.parametrize(
    "mutation",
    [
        "digest",
        "record",
        "status",
        "run_id",
        "path",
        "missing_role",
        "schema",
        "report",
    ],
)
def test_pilot_rejects_tampered_or_incomplete_evidence(bundle, mutation):
    path = bundle / "verification.json"
    data = json.loads(path.read_text())
    if mutation == "digest":
        data["records"]["candidate"]["record_digest"] = "a" * 64
    elif mutation == "record":
        rp = bundle / "candidate.run.json"
        record = json.loads(rp.read_text())
        record["record"]["run_id"] = "changed"
        rp.write_text(json.dumps(record))
    elif mutation == "status":
        data["records"]["candidate"]["status"] = "failed"
    elif mutation == "run_id":
        data["records"]["candidate"]["run_id"] = "other"
    elif mutation == "path":
        data["records"]["candidate"]["path"] = "../candidate.run.json"
    elif mutation == "missing_role":
        del data["records"]["candidate"]
    elif mutation == "schema":
        data["schema"] = "other"
    elif mutation == "report":
        (bundle / "report.md").write_text("changed report")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        record_pilot(bundle, notes())


def test_cli_retains_notes_and_refuses_to_overwrite(bundle, tmp_path):
    note_path, output = tmp_path / "notes.json", tmp_path / "pilot.json"
    note_path.write_text(json.dumps(notes()))
    args = [
        "pilot",
        "record",
        "--evidence",
        str(bundle),
        "--notes",
        str(note_path),
        "--output",
        str(output),
    ]
    stdout, stderr = StringIO(), StringIO()
    assert main(args, stdout=stdout, stderr=stderr) == 0, stderr.getvalue()
    original = output.read_bytes()
    assert main(args, stdout=stdout, stderr=stderr) == 2
    assert output.read_bytes() == original
    assert "Decision: inconclusive" in stdout.getvalue()


def test_negative_verification_result_is_retained_as_negative_pilot_evidence(
    prefix_case, tmp_path
):
    prefix_case["mode"] = "inference_failed"
    result = execute(prefix_case, tmp_path / "failed")
    receipt = record_pilot(result.output_dir, {**notes(), "decision": "reject"})
    assert receipt["evidence"]["verification_verdict"] == "EXECUTION_FAILED"
    assert receipt["notes"]["decision"] == "reject"


def test_bundle_file_size_is_bounded(tmp_path):
    with (tmp_path / "verification.json").open("wb") as stream:
        stream.seek(32 * 1024 * 1024)
        stream.write(b"x")
    with pytest.raises(ValueError, match="32 MiB"):
        record_pilot(tmp_path, notes())


@pytest.mark.parametrize("field", ["scope", "analyses"])
def test_malformed_report_fields_are_validation_errors(bundle, field):
    path = bundle / "verification.json"
    data = json.loads(path.read_text())
    del data[field]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        record_pilot(bundle, notes())


def test_symlinked_evidence_cannot_escape_bundle(bundle, tmp_path):
    path = bundle / "report.md"
    external = tmp_path / "outside.md"
    external.write_bytes(path.read_bytes())
    path.unlink()
    try:
        path.symlink_to(external)
    except OSError:
        pytest.skip("symlink creation unavailable on this host")
    with pytest.raises(ValueError, match="within"):
        record_pilot(bundle, notes())
