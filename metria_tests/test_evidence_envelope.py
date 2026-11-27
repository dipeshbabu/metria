"""Signed, challenge-bound collector evidence envelopes."""

from __future__ import annotations

import base64
import copy
from datetime import datetime, timedelta, timezone

import pytest

from metria.evidence_envelope import (
    Challenge,
    EnvelopeError,
    KeyRing,
    NonceLedger,
    TrustedKey,
    TrustPolicy,
    canonical_bytes,
    sign,
    verify_envelope,
)

NOW = datetime(2026, 11, 27, 12, 0, tzinfo=timezone.utc)
KEY = TrustedKey("collector-2026-11", "hmac-sha256", b"k" * 32)
RING = KeyRing({KEY.key_id: KEY})
CHALLENGE = Challenge("nonce-0123456789abcdef", "run-0001", "candidate", "fp8-pool")


def _payload(**changes):
    payload = {
        "collector": {"name": "metria-collector", "version": "0.1.8"},
        "challenge": CHALLENGE.as_mapping(),
        "collected_at": NOW.isoformat(),
        "facts": {
            "engine.kv_cache_dtype": "fp8",
            "engine.tensor_parallel_size": 1,
            "gpu.devices": [{"name": "NVIDIA L4", "uuid_sha256": "a" * 64}],
        },
        "unavailable": ["image.digest"],
    }
    payload.update(changes)
    return payload


def _verify(envelope, **policy):
    return verify_envelope(
        envelope,
        keyring=RING,
        challenge=CHALLENGE,
        policy=TrustPolicy(**policy),
        now=NOW + timedelta(seconds=5),
    )


def test_valid_envelope_yields_collector_observed_facts():
    verified = _verify(sign(_payload(), KEY))
    assert verified["authority"] == "collector_observed"
    assert verified["facts"]["engine.kv_cache_dtype"] == "fp8"
    assert verified["unavailable"] == ["image.digest"]
    assert len(verified["envelope_sha256"]) == 64


def test_canonical_bytes_are_order_independent():
    assert canonical_bytes({"b": 1, "a": [1, {"d": 2, "c": 3}]}) == canonical_bytes(
        {"a": [1, {"c": 3, "d": 2}], "b": 1}
    )
    with pytest.raises(ValueError):
        canonical_bytes({"x": float("nan")})


def _tamper(envelope, mutate):
    changed = copy.deepcopy(envelope)
    mutate(changed)
    return changed


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda e: e["payload"]["facts"].update({"engine.kv_cache_dtype": "auto"}),
            "signature is invalid",
        ),
        (
            lambda e: e.update(signature=base64.b64encode(b"x" * 32).decode()),
            "signature is invalid",
        ),
        (lambda e: e.update(signature="***"), "not base64"),
        (lambda e: e.update(key_id="unknown"), "not trusted"),
        (lambda e: e.update(algorithm="ed25519"), "does not match"),
        (lambda e: e.update(schema="other"), "schema"),
        (lambda e: e.pop("signature"), "unexpected fields"),
    ],
)
def test_tampering_and_unknown_keys_are_rejected(mutate, match):
    with pytest.raises(EnvelopeError, match=match):
        _verify(_tamper(sign(_payload(), KEY), mutate))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("nonce", "nonce-ffffffffffffffff"),
        ("run_id", "run-0002"),
        ("role", "reference"),
        ("deployment", "auto-pool"),
    ],
)
def test_envelope_bound_to_another_challenge_is_rejected(field, value):
    challenge = {**CHALLENGE.as_mapping(), field: value}
    with pytest.raises(EnvelopeError, match="different nonce, run, role or deployment"):
        _verify(sign(_payload(challenge=challenge), KEY))


def test_stale_and_future_envelopes_are_rejected():
    old = (NOW - timedelta(minutes=10)).isoformat()
    with pytest.raises(EnvelopeError, match="stale"):
        _verify(sign(_payload(collected_at=old), KEY))
    future = (NOW + timedelta(minutes=5)).isoformat()
    with pytest.raises(EnvelopeError, match="future"):
        _verify(sign(_payload(collected_at=future), KEY))
    assert _verify(sign(_payload(collected_at=old), KEY), max_age_s=3600)


def test_required_facts_and_allowed_keys():
    envelope = sign(_payload(), KEY)
    with pytest.raises(
        EnvelopeError, match="required facts are unavailable: image.digest"
    ):
        _verify(envelope, require_collector=True, required_facts=("image.digest",))
    with pytest.raises(EnvelopeError, match="not allowed"):
        _verify(envelope, allowed_key_ids=("collector-2027-01",))
    assert _verify(envelope, allowed_key_ids=(KEY.key_id,))


@pytest.mark.parametrize(
    ("facts", "match"),
    [
        ({"model.path": "/home/alice/models/llama"}, "private home path"),
        ({"env": {"HF_TOKEN": "x"}}, "looks like a secret"),
        ({"engine": {"api_key": "x"}}, "looks like a secret"),
        ({"x": float("inf")}, "not finite"),
    ],
)
def test_collectors_cannot_sign_unredacted_facts(facts, match):
    with pytest.raises(EnvelopeError, match=match):
        sign(_payload(facts=facts), KEY)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"facts": {"a": 1}, "unavailable": ["a"]}, "both present and unavailable"),
        ({"challenge": {**CHALLENGE.as_mapping(), "nonce": "short"}}, "nonce"),
        ({"challenge": {**CHALLENGE.as_mapping(), "role": "judge"}}, "role"),
        ({"collected_at": "2026-11-27T12:00:00"}, "timezone"),
        ({"unavailable": [1]}, "list fact names"),
    ],
)
def test_payload_shape_is_validated(changes, match):
    with pytest.raises(EnvelopeError, match=match):
        sign(_payload(**changes), KEY)


def test_nonce_ledger_rejects_replay():
    ledger = NonceLedger()
    ledger.consume(["nonce-a" * 3])
    with pytest.raises(EnvelopeError, match="replay"):
        ledger.consume(["nonce-a" * 3])


def test_key_ring_parsing_and_key_validation():
    ring = KeyRing.from_mapping(
        {
            "k1": {
                "algorithm": "hmac-sha256",
                "key": base64.b64encode(b"s" * 32).decode(),
            }
        }
    )
    assert ring.keys["k1"].material == b"s" * 32
    with pytest.raises(EnvelopeError, match="at least 32 bytes"):
        TrustedKey("short", "hmac-sha256", b"x")
    with pytest.raises(EnvelopeError, match="unsupported"):
        TrustedKey("k", "rsa", b"x" * 32)
    with pytest.raises(EnvelopeError, match="not base64"):
        KeyRing.from_mapping({"k": {"algorithm": "hmac-sha256", "key": "***"}})


def test_trust_policy_validation():
    assert TrustPolicy.from_mapping(None) == TrustPolicy()
    with pytest.raises(EnvelopeError, match="accepts only"):
        TrustPolicy.from_mapping({"trust_everything": True})
    with pytest.raises(EnvelopeError, match="need require_collector"):
        TrustPolicy.from_mapping({"required_facts": ["engine.kv_cache_dtype"]})
    with pytest.raises(EnvelopeError, match="max_age_s"):
        TrustPolicy.from_mapping({"max_age_s": 0})


def test_ed25519_envelopes_verify_when_cryptography_is_available():
    ed25519 = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    from cryptography.hazmat.primitives import serialization

    private = ed25519.Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    key = TrustedKey("ed-1", "ed25519", public)
    payload = _payload()
    envelope = {
        "schema": "metria.endpoint_evidence.v1",
        "key_id": "ed-1",
        "algorithm": "ed25519",
        "payload": payload,
        "signature": base64.b64encode(private.sign(canonical_bytes(payload))).decode(),
    }
    verified = verify_envelope(
        envelope,
        keyring=KeyRing({"ed-1": key}),
        challenge=CHALLENGE,
        policy=TrustPolicy(),
        now=NOW,
    )
    assert verified["algorithm"] == "ed25519"
