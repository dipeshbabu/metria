"""Signed, challenge-bound evidence envelopes from optional host collectors.

A remote endpoint can only *claim* what it runs. A collector process on the
serving host can read more (engine configuration, model file digests, GPUs,
driver and numerics settings) and return it in a versioned envelope that is
signed over canonical bytes and bound to the verifier's challenge: a fresh
nonce, the run ID, the role (reference or candidate) and the declared
deployment, inside a freshness window.

Authority levels are explicit and never inferred:

``operator_declared``
    What the recipe author stated.
``server_claimed``
    What the inference server reported about itself.
``collector_observed``
    Read by a collector whose signature verified against a trusted key.
``hardware_attested``
    Reserved for hardware-rooted attestation; no implementation yet.

A valid signature proves the envelope came from a holder of a trusted key and
was not modified. It does not prove the facts are true on a compromised host.

The core stays dependency-free: HMAC-SHA256 uses the standard library. Ed25519
public-key verification is available when the optional ``cryptography``
package is installed.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

SCHEMA = "metria.endpoint_evidence.v1"
AUTHORITIES = (
    "operator_declared",
    "server_claimed",
    "collector_observed",
    "hardware_attested",
)
ALGORITHMS = frozenset({"hmac-sha256", "ed25519"})
_PAYLOAD_KEYS = frozenset(
    {"collector", "challenge", "collected_at", "facts", "unavailable"}
)
_CHALLENGE_KEYS = ("nonce", "run_id", "role", "deployment")
_NONCE = re.compile(r"[A-Za-z0-9_-]{16,128}")
# Values that must never leave a collector: home paths and secret-looking keys.
_HOME_PATH = re.compile(r"(?:[A-Za-z]:[\\/]+Users[\\/]|/Users/|/home/|/root/)")
_SECRET_KEY = re.compile(
    r"(?:^|_)(?:API_?KEY|KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?|AUTH)(?:_|$)",
    re.IGNORECASE,
)


class EnvelopeError(ValueError):
    """An envelope is malformed, unverifiable, stale or bound to another run."""


def canonical_bytes(value: Any) -> bytes:
    """Deterministic UTF-8 JSON: sorted keys, no whitespace, no NaN."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise EnvelopeError(f"{label} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise EnvelopeError(f"{label} must be an ISO-8601 UTC timestamp") from None
    if parsed.tzinfo is None:
        raise EnvelopeError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class TrustedKey:
    """One verification key. HMAC keys are shared secrets; Ed25519 keys are public."""

    key_id: str
    algorithm: str
    material: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", self.key_id):
            raise EnvelopeError("key IDs use 1-64 letters, digits, '.', '_' or '-'")
        if self.algorithm not in ALGORITHMS:
            raise EnvelopeError(f"unsupported signature algorithm {self.algorithm!r}")
        minimum = 32 if self.algorithm == "hmac-sha256" else 32
        if len(self.material) < minimum:
            raise EnvelopeError(f"{self.algorithm} keys need at least {minimum} bytes")


@dataclass(frozen=True)
class KeyRing:
    """Trusted keys by ID; rotating a key means adding a new ID and retiring the old."""

    keys: Mapping[str, TrustedKey]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> KeyRing:
        """Parse ``{key_id: {"algorithm": ..., "key": base64}}``."""

        keys = {}
        for key_id, entry in value.items():
            if not isinstance(entry, Mapping) or set(entry) != {"algorithm", "key"}:
                raise EnvelopeError("each trusted key needs exactly algorithm and key")
            try:
                material = base64.b64decode(entry["key"], validate=True)
            except (ValueError, TypeError):
                raise EnvelopeError(f"trusted key {key_id!r} is not base64") from None
            keys[key_id] = TrustedKey(key_id, entry["algorithm"], material)
        return cls(keys)


def sign(payload: Mapping[str, Any], key: TrustedKey) -> dict[str, Any]:
    """Sign canonical payload bytes. Ed25519 signing needs a private key: use HMAC here."""

    if key.algorithm != "hmac-sha256":
        raise EnvelopeError("this helper signs with HMAC keys only")
    validate_payload(payload)
    digest = hmac.new(key.material, canonical_bytes(payload), hashlib.sha256)
    return {
        "schema": SCHEMA,
        "key_id": key.key_id,
        "algorithm": key.algorithm,
        "payload": dict(payload),
        "signature": base64.b64encode(digest.digest()).decode("ascii"),
    }


def _verify_signature(key: TrustedKey, message: bytes, signature: bytes) -> bool:
    if key.algorithm == "hmac-sha256":
        expected = hmac.new(key.material, message, hashlib.sha256).digest()
        return hmac.compare_digest(expected, signature)
    try:
        from cryptography.exceptions import (  # type: ignore[import-not-found,unused-ignore]
            InvalidSignature,
        )
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # type: ignore[import-not-found,unused-ignore]
            Ed25519PublicKey,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise EnvelopeError(
            "Ed25519 verification needs the optional 'cryptography' package"
        ) from exc
    try:
        Ed25519PublicKey.from_public_bytes(key.material).verify(signature, message)
    except (InvalidSignature, ValueError):
        return False
    return True


def _check_redaction(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _SECRET_KEY.search(str(key)):
                raise EnvelopeError(f"evidence field {path}.{key} looks like a secret")
            _check_redaction(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _check_redaction(item, f"{path}[{index}]")
    elif isinstance(value, str) and _HOME_PATH.search(value):
        raise EnvelopeError(f"evidence field {path} contains a private home path")
    elif isinstance(value, float) and not math.isfinite(value):
        raise EnvelopeError(f"evidence field {path} is not finite")


def validate_payload(payload: Mapping[str, Any]) -> None:
    """Shape and redaction rules shared by signers and verifiers."""

    if not isinstance(payload, Mapping) or set(payload) != _PAYLOAD_KEYS:
        raise EnvelopeError(
            "payload needs collector, challenge, collected_at, facts and unavailable"
        )
    challenge = payload["challenge"]
    if not isinstance(challenge, Mapping) or set(challenge) != set(_CHALLENGE_KEYS):
        raise EnvelopeError("challenge needs nonce, run_id, role and deployment")
    if not isinstance(challenge["nonce"], str) or not _NONCE.fullmatch(
        challenge["nonce"]
    ):
        raise EnvelopeError("challenge nonce must be 16-128 URL-safe characters")
    if challenge["role"] not in {"reference", "candidate"}:
        raise EnvelopeError("challenge role must be reference or candidate")
    _timestamp(payload["collected_at"], "collected_at")
    if not isinstance(payload["facts"], Mapping):
        raise EnvelopeError("facts must be an object")
    unavailable = payload["unavailable"]
    if not isinstance(unavailable, list) or any(
        not isinstance(i, str) for i in unavailable
    ):
        raise EnvelopeError("unavailable must list fact names")
    if set(unavailable) & set(payload["facts"]):
        raise EnvelopeError("a fact cannot be both present and unavailable")
    _check_redaction(payload["facts"], "facts")


@dataclass(frozen=True)
class Challenge:
    """What the verifier asked for; the envelope must echo it exactly."""

    nonce: str
    run_id: str
    role: str
    deployment: str

    def as_mapping(self) -> dict[str, str]:
        return {key: getattr(self, key) for key in _CHALLENGE_KEYS}


@dataclass(frozen=True)
class TrustPolicy:
    """Opt-in requirements for collector evidence."""

    require_collector: bool = False
    required_facts: tuple[str, ...] = ()
    max_age_s: float = 300.0
    max_skew_s: float = 30.0
    allowed_key_ids: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> TrustPolicy:
        if value is None:
            return cls()
        allowed = {
            "require_collector",
            "required_facts",
            "max_age_s",
            "max_skew_s",
            "allowed_key_ids",
        }
        if not isinstance(value, Mapping) or set(value) - allowed:
            raise EnvelopeError(
                "trust policy accepts only: " + ", ".join(sorted(allowed))
            )
        policy = cls(
            require_collector=bool(value.get("require_collector", False)),
            required_facts=tuple(value.get("required_facts", ())),
            max_age_s=float(value.get("max_age_s", 300.0)),
            max_skew_s=float(value.get("max_skew_s", 30.0)),
            allowed_key_ids=tuple(value.get("allowed_key_ids", ())),
        )
        if not 0 < policy.max_age_s <= 86400 or not 0 <= policy.max_skew_s <= 600:
            raise EnvelopeError(
                "max_age_s must be in (0, 86400] and max_skew_s in [0, 600]"
            )
        if policy.required_facts and not policy.require_collector:
            raise EnvelopeError("required_facts need require_collector")
        return policy


def verify_envelope(
    envelope: Mapping[str, Any],
    *,
    keyring: KeyRing,
    challenge: Challenge,
    policy: TrustPolicy,
    now: datetime,
) -> dict[str, Any]:
    """Return the verified facts, or raise ``EnvelopeError`` with the first reason."""

    if not isinstance(envelope, Mapping) or envelope.get("schema") != SCHEMA:
        raise EnvelopeError(f"envelope schema must be {SCHEMA}")
    if set(envelope) != {"schema", "key_id", "algorithm", "payload", "signature"}:
        raise EnvelopeError("envelope has missing or unexpected fields")
    key = keyring.keys.get(envelope["key_id"])
    if key is None:
        raise EnvelopeError("envelope key ID is not trusted")
    if policy.allowed_key_ids and key.key_id not in policy.allowed_key_ids:
        raise EnvelopeError("envelope key ID is not allowed by the trust policy")
    if envelope["algorithm"] != key.algorithm:
        raise EnvelopeError("envelope algorithm does not match the trusted key")
    payload = envelope["payload"]
    validate_payload(payload)
    try:
        signature = base64.b64decode(envelope["signature"], validate=True)
    except (ValueError, TypeError):
        raise EnvelopeError("envelope signature is not base64") from None
    if not _verify_signature(key, canonical_bytes(payload), signature):
        raise EnvelopeError("envelope signature is invalid")
    if dict(payload["challenge"]) != challenge.as_mapping():
        raise EnvelopeError(
            "envelope is bound to a different nonce, run, role or deployment"
        )
    collected = _timestamp(payload["collected_at"], "collected_at")
    age = (now - collected).total_seconds()
    if age > policy.max_age_s:
        raise EnvelopeError("envelope is stale")
    if age < -policy.max_skew_s:
        raise EnvelopeError("envelope is dated in the future")
    missing = [name for name in policy.required_facts if name not in payload["facts"]]
    if missing:
        raise EnvelopeError("required facts are unavailable: " + ", ".join(missing))
    return {
        "authority": "collector_observed",
        "key_id": key.key_id,
        "algorithm": key.algorithm,
        "collector": payload["collector"],
        "collected_at": payload["collected_at"],
        "facts": dict(payload["facts"]),
        "unavailable": list(payload["unavailable"]),
        "envelope_sha256": hashlib.sha256(canonical_bytes(envelope)).hexdigest(),
    }


class NonceLedger:
    """Reject a nonce seen before within one verification (replay protection)."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def consume(self, nonces: Iterable[str]) -> None:
        for nonce in nonces:
            if nonce in self._seen:
                raise EnvelopeError("envelope nonce was already used (replay)")
            self._seen.add(nonce)
