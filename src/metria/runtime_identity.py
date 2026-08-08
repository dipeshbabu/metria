"""Shared observed model/tokenizer identity evidence for Metria runtimes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any


class IdentityStatus(str, Enum):
    """Confidence/result of comparing requested and observed runtime identity."""

    MATCHED = "matched"
    PARTIAL = "partial"
    UNVERIFIED = "unverified"
    MISMATCH = "mismatch"


def _nonempty_text(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _requested_identity(model: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    aliases = {
        "id": "id",
        "revision": "revision",
        "tokenizer_id": "tokenizer_id",
        "tokenizer_revision": "tokenizer_revision",
        "path": "path",
    }
    for source, target in aliases.items():
        value = _nonempty_text(model.get(source))
        if value is not None:
            result[target] = value
    return result


def _observed_text(mapping: Mapping[str, Any] | None, *names: str) -> str | None:
    if mapping is None:
        return None
    for name in names:
        value = _nonempty_text(mapping.get(name))
        if value is not None:
            return value
    return None


def build_runtime_identity_evidence(
    requested_model: Mapping[str, Any],
    *,
    observed_model: Mapping[str, Any] | None = None,
    observed_tokenizer: Mapping[str, Any] | None = None,
    source: str,
    additional_evidence: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Compare only identity fields a runtime independently exposes.

    Missing observed fields remain unverified. Requested values are never copied
    into the observed side merely to make the identity appear matched.
    """

    requested = _requested_identity(requested_model)
    model_requested = {
        key: value for key, value in requested.items() if key in {"id", "revision", "path"}
    }
    tokenizer_requested = {
        key.removeprefix("tokenizer_"): value
        for key, value in requested.items()
        if key in {"tokenizer_id", "tokenizer_revision"}
    }

    normalized_model: dict[str, Any] = {}
    model_id = _observed_text(observed_model, "id", "model", "name_or_path")
    model_revision = _observed_text(observed_model, "revision", "commit_hash")
    model_path = _observed_text(observed_model, "path")
    if model_id is not None:
        normalized_model["id"] = model_id
    if model_revision is not None:
        normalized_model["revision"] = model_revision
    if model_path is not None:
        normalized_model["path"] = model_path
    if observed_model is not None:
        for key in ("sha256", "size_bytes"):
            if key in observed_model:
                normalized_model[key] = observed_model[key]

    normalized_tokenizer: dict[str, Any] = {}
    tokenizer_id = _observed_text(observed_tokenizer, "id", "name_or_path", "tokenizer")
    tokenizer_revision = _observed_text(
        observed_tokenizer,
        "revision",
        "commit_hash",
        "tokenizer_revision",
    )
    if tokenizer_id is not None:
        normalized_tokenizer["id"] = tokenizer_id
    if tokenizer_revision is not None:
        normalized_tokenizer["revision"] = tokenizer_revision

    mismatches: list[str] = []
    verified_fields: list[str] = []

    for field in ("id", "revision", "path"):
        requested_value = model_requested.get(field)
        observed_value = normalized_model.get(field)
        if requested_value is None or observed_value is None:
            continue
        if requested_value == observed_value:
            verified_fields.append(f"model.{field}")
        else:
            mismatches.append(f"model.{field}")

    for field in ("id", "revision"):
        requested_value = tokenizer_requested.get(field)
        observed_value = normalized_tokenizer.get(field)
        if requested_value is None or observed_value is None:
            continue
        if requested_value == observed_value:
            verified_fields.append(f"tokenizer.{field}")
        else:
            mismatches.append(f"tokenizer.{field}")

    requested_comparable = len(model_requested) + len(tokenizer_requested)
    observed_comparable = sum(
        key in normalized_model for key in ("id", "revision", "path")
    ) + sum(key in normalized_tokenizer for key in ("id", "revision"))

    if mismatches:
        status = IdentityStatus.MISMATCH
    elif not verified_fields and observed_comparable == 0:
        status = IdentityStatus.UNVERIFIED
    elif len(verified_fields) == requested_comparable and requested_comparable > 0:
        status = IdentityStatus.MATCHED
    else:
        status = IdentityStatus.PARTIAL

    return {
        "status": status.value,
        "source": source,
        "model": {
            "requested": model_requested,
            "observed": normalized_model,
        },
        "tokenizer": {
            "requested": tokenizer_requested,
            "observed": normalized_tokenizer,
        },
        "verified_fields": tuple(verified_fields),
        "mismatches": tuple(mismatches),
        "evidence": dict(additional_evidence or {}),
    }


def runtime_identity_mismatches(observed: Mapping[str, Any]) -> tuple[str, ...]:
    """Return declared identity contradictions from a runtime observation."""

    identity = observed.get("identity")
    if not isinstance(identity, Mapping):
        return ()
    raw = identity.get("mismatches", ())
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        return ()
    return tuple(item for item in raw if isinstance(item, str) and item)
