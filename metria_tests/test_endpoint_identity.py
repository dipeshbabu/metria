"""Claimed endpoint identity: snapshots, declared contradictions, tokenizer check."""

from __future__ import annotations

import pytest
from metria_tests.test_endpoint_runtime import (  # noqa: F401
    KEY,
    _run,
    _spec,
    server,
)

from metria import RunSpec
from metria.identity_evidence import RuntimeIdentityEvidence
from metria.records import run_record_to_json
from metria.runtimes.endpoint_identity import (
    SNAPSHOT_GAP,
    TOKENIZER_GAP,
    identity_gaps,
    validate_declared,
)


def _declared(spec: RunSpec, **declared) -> RunSpec:
    return RunSpec(
        **{**spec.__dict__, "runtime": {**spec.runtime, "declared": declared}}
    )


def test_identity_is_claimed_partial_and_stable(server, tmp_path):  # noqa: F811
    record = _run(_declared(_spec(tmp_path), engine_version="0.30.0", hardware="L4"))
    assert record.status.value == "completed", record.events
    observed = record.observed
    identity = observed["identity"]
    assert identity["status"] == "partial"
    assert identity["runtime"]["version"] == "0.30.0"
    assert identity["runtime"]["source"] == "server_reported"
    assert identity["model"]["root"] == "org/model"
    assert identity["tokenizer"]["status"] == "partial"
    assert identity["tokenizer"]["checked_prompts"] == 2
    assert observed["identity_authority"] == "claimed"
    assert observed["identity_snapshots"]["stable"] is True
    assert observed["declared"] == {"engine_version": "0.30.0", "hardware": "L4"}
    assert identity_gaps(observed) == []
    assert KEY not in run_record_to_json(record)


def test_identity_drift_is_an_evidence_gap(server, tmp_path):  # noqa: F811
    server.drift_after = 1
    record = _run(_spec(tmp_path))
    assert record.status.value == "completed"
    snapshots = record.observed["identity_snapshots"]
    assert snapshots["before"]["runtime"]["version"] == "0.30.0"
    assert snapshots["after"]["runtime"]["version"] == "0.31.0"
    assert identity_gaps(record.observed) == [SNAPSHOT_GAP]


def test_declared_contradiction_fails_before_measurement(server, tmp_path):  # noqa: F811
    record = _run(_declared(_spec(tmp_path), engine_version="0.29.0"))
    assert record.status.value == "failed"
    assert any(event.get("stage") == "launch" for event in record.events)
    assert server.completions == 0


def test_declared_max_model_len_is_checked(server, tmp_path):  # noqa: F811
    assert _run(_declared(_spec(tmp_path), max_model_len=1024)).status.value == "failed"


def test_tokenizer_disagreement_is_a_mismatch_gap(server, tmp_path):  # noqa: F811
    server.tokenize_offset = 1
    record = _run(_spec(tmp_path))
    assert record.status.value == "completed"
    tokenizer = record.observed["identity"]["tokenizer"]
    assert tokenizer["status"] == "mismatch" and tokenizer["mismatched_prompts"] == 2
    assert record.observed["identity"]["status"] == "mismatch"
    assert TOKENIZER_GAP in identity_gaps(record.observed)


def test_unexposed_tokenize_stays_unknown_without_a_gap(server, tmp_path):  # noqa: F811
    server.tokenize_status = 404
    record = _run(_spec(tmp_path))
    assert record.status.value == "completed"
    assert record.observed["identity"]["tokenizer"]["status"] == "unknown"
    assert identity_gaps(record.observed) == []


@pytest.mark.parametrize(
    ("declared", "match"),
    [
        ({"api_key": "x"}, "accepts only"),
        ({"hardware": ["L4"]}, "string or integer"),
        ({"hardware": True}, "string or integer"),
        ({"hardware": ""}, "1-256"),
    ],
)
def test_declared_block_is_allowlisted(declared, match):
    with pytest.raises(ValueError, match=match):
        validate_declared(declared)


def test_unsafe_endpoint_fields_are_rejected_by_identity():
    with pytest.raises(ValueError, match="sensitive"):
        RuntimeIdentityEvidence(
            status="unknown", endpoint={"url": "https://x", "authorization": "k"}
        )


def test_filesystem_model_roots_are_hashed_not_retained(server, tmp_path):  # noqa: F811
    import hashlib

    private = "/srv/models/private-checkout"
    server.root = private
    record = _run(_declared(_spec(tmp_path), model_root=private))
    assert record.status.value == "completed", record.events
    model = record.observed["identity"]["model"]
    assert "root" not in model
    assert model["root_sha256"] == hashlib.sha256(private.encode()).hexdigest()
    # The server's claim is hashed; only the operator's own declaration names it.
    assert private not in str(dict(record.observed["identity"]))
    assert private not in str(dict(record.observed["identity_snapshots"]))
    assert (
        _run(_declared(_spec(tmp_path), model_root="/other")).status.value == "failed"
    )
