"""Challenge an optional host evidence collector from an endpoint run.

Each endpoint run may name a collector (see ``metria collector serve``). The
adapter challenges it at launch and again after the last request, each time
with a fresh nonce bound to the run's session, role and declared deployment,
and verifies the signed envelope against trusted keys. Verified facts are
``collector_observed``; failures are retained as reasons, never as facts.

Comparable facts drop per-host identifiers (GPU UUID hashes, the model path
hash, which GPU index a server uses) so two servers on different hosts can
still be compared; those identifiers stay in the run's receipts.
"""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from ..evidence_envelope import (
    Challenge,
    EnvelopeError,
    KeyRing,
    NonceLedger,
    TrustPolicy,
    canonical_bytes,
    verify_envelope,
)
from .endpoint_http import EndpointConnection, EndpointError, _env_name

COLLECTOR_KEYS = frozenset({"url_env", "keys_env", "deployment", "role", "trust"})
_ROLES = frozenset({"reference", "candidate"})
# Declared deployment facts the collector can corroborate.
DECLARED_TO_FACT = {
    "kv_cache_dtype": "engine.kv_cache_dtype",
    "dtype": "engine.dtype",
    "tensor_parallel_size": "engine.tensor_parallel_size",
    "pipeline_parallel_size": "engine.pipeline_parallel_size",
    "max_model_len": "engine.max_model_len",
    "model_revision": "model.revision",
}
_PLACEMENT_VARIABLES = frozenset({"CUDA_VISIBLE_DEVICES"})


def validate_collector(value: Any) -> dict[str, Any] | None:
    """Validate a run's ``runtime.collector`` block; ``None`` means claim-only."""

    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) - COLLECTOR_KEYS:
        raise ValueError(
            "runtime.collector accepts only: " + ", ".join(sorted(COLLECTOR_KEYS))
        )
    if not {"url_env", "keys_env", "deployment", "role"} <= set(value):
        raise ValueError(
            "runtime.collector needs url_env, keys_env, deployment and role"
        )
    _env_name(value["url_env"], "collector.url_env")
    _env_name(value["keys_env"], "collector.keys_env")
    deployment = value["deployment"]
    if not isinstance(deployment, str) or not 0 < len(deployment) <= 128:
        raise ValueError("collector.deployment must be a 1-128 character name")
    if value["role"] not in _ROLES:
        raise ValueError("collector.role must be reference or candidate")
    try:
        TrustPolicy.from_mapping(value.get("trust"))
    except EnvelopeError as exc:
        raise ValueError(str(exc)) from None
    return {
        "url_env": value["url_env"],
        "keys_env": value["keys_env"],
        "deployment": deployment,
        "role": value["role"],
        "trust": dict(value.get("trust") or {}),
    }


def comparable_facts(facts: Mapping[str, Any]) -> dict[str, Any]:
    """Facts that should match across hosts unless a change is declared."""

    result = {key: value for key, value in facts.items() if key != "model.path_sha256"}
    devices = result.get("gpu.devices")
    if isinstance(devices, list):
        result["gpu.devices"] = [
            {k: v for k, v in device.items() if k not in {"uuid_sha256", "index"}}
            for device in devices
            if isinstance(device, Mapping)
        ]
    numerics = result.get("environment.numerics")
    if isinstance(numerics, Mapping):
        result["environment.numerics"] = {
            k: v for k, v in numerics.items() if k not in _PLACEMENT_VARIABLES
        }
    return result


class CollectorSession:
    """Challenges one collector for one run; holds receipts and verdicts."""

    def __init__(self, config: Mapping[str, Any], timeout_s: float) -> None:
        url = os.environ.get(config["url_env"])
        if not url:
            raise ValueError(f"environment variable {config['url_env']} is not set")
        raw_keys = os.environ.get(config["keys_env"])
        if not raw_keys:
            raise ValueError(f"environment variable {config['keys_env']} is not set")
        try:
            self.keyring = KeyRing.from_mapping(json.loads(raw_keys))
        except (ValueError, EnvelopeError) as exc:
            raise ValueError(f"{config['keys_env']} must hold a JSON key ring") from exc
        from .endpoint_http import parse_base_url

        self.connection = EndpointConnection(parse_base_url(url), None, timeout_s, 1)
        self.config = config
        self.policy = TrustPolicy.from_mapping(config.get("trust"))
        self.session_id = secrets.token_urlsafe(12)
        self.ledger = NonceLedger()
        self.snapshots: dict[str, dict[str, Any]] = {}

    def snapshot(self, name: str, now: datetime | None = None) -> dict[str, Any]:
        """Challenge the collector once and verify the reply."""

        challenge = Challenge(
            nonce=secrets.token_urlsafe(24),
            run_id=self.session_id,
            role=self.config["role"],
            deployment=self.config["deployment"],
        )
        record: dict[str, Any] = {"verified": False}
        try:
            self.ledger.consume([challenge.nonce])
            envelope, _ = self.connection.request(
                "POST", "/evidence", challenge.as_mapping()
            )
            verified = verify_envelope(
                envelope,
                keyring=self.keyring,
                challenge=challenge,
                policy=self.policy,
                now=now or datetime.now(timezone.utc),
            )
        except (EnvelopeError, EndpointError, ValueError) as exc:
            record["error"] = str(exc)
        else:
            record.update(verified=True, **verified)
        self.snapshots[name] = record
        return record

    def observed(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Compared evidence and non-compared receipts for the run record."""

        before = self.snapshots.get("before", {"verified": False})
        after = self.snapshots.get("after", {"verified": False})
        both = before.get("verified") and after.get("verified")
        facts = comparable_facts(before.get("facts", {})) if both else {}
        stable = bool(both) and canonical_bytes(
            comparable_facts(before["facts"])
        ) == canonical_bytes(comparable_facts(after["facts"]))
        compared = {
            "authority": "collector_observed" if both else "unavailable",
            "verified": bool(both),
            "stable": stable,
            "facts": facts,
            "unavailable": sorted(before.get("unavailable", [])) if both else [],
            "errors": [r["error"] for r in (before, after) if r.get("error")],
        }
        receipts = {
            "session_id": self.session_id,
            "deployment": self.config["deployment"],
            "snapshots": {
                name: {
                    key: record.get(key)
                    for key in (
                        "verified",
                        "error",
                        "key_id",
                        "collected_at",
                        "envelope_sha256",
                    )
                    if key in record
                }
                for name, record in self.snapshots.items()
            },
            "facts": before.get("facts", {}),
        }
        return compared, receipts
