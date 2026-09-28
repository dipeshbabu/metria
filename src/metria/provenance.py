"""Versioned serialization and publication checks for shared artifact manifests."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from .artifacts import artifact_to_data
from .identity import ArtifactManifest

ARTIFACT_MANIFEST_SCHEMA = "metria.artifact_manifest.v1"
_FIELDS = {
    "name",
    "kind",
    "uri",
    "path",
    "revision",
    "sha256",
    "size_bytes",
    "source",
    "metadata",
}


def artifact_manifest_to_data(manifest: ArtifactManifest) -> dict[str, Any]:
    """Wrap the existing artifact primitive in its versioned wire envelope."""
    return {"schema": ARTIFACT_MANIFEST_SCHEMA, "artifact": artifact_to_data(manifest)}


def artifact_manifest_from_data(data: Mapping[str, Any]) -> ArtifactManifest:
    """Reject unknown envelope fields and reconstruct the shared primitive."""
    if not isinstance(data, Mapping) or set(data) != {"schema", "artifact"}:
        raise ValueError("artifact manifest requires exactly schema and artifact")
    if data["schema"] != ARTIFACT_MANIFEST_SCHEMA:
        raise ValueError("unsupported artifact manifest schema")
    artifact = data["artifact"]
    if not isinstance(artifact, Mapping) or set(artifact) - _FIELDS:
        raise ValueError("artifact manifest contains invalid or unknown fields")
    if not {"name", "kind"} <= set(artifact):
        raise ValueError("artifact manifest requires name and kind")
    # Apply the same strict JSON value rules as run and recipe serialization.
    manifest = ArtifactManifest(**dict(artifact))
    artifact_to_data(manifest)
    return manifest


def artifact_manifest_to_json(manifest: ArtifactManifest) -> str:
    return json.dumps(
        artifact_manifest_to_data(manifest), sort_keys=True, indent=2, allow_nan=False
    )


def validate_headline_manifest(manifest: ArtifactManifest) -> None:
    """Require evidence and rights metadata before designating a headline result.

    This validates presence and shape, not the truth of a provenance claim or
    permission to redistribute an upstream input. Explicit unknown licensing
    remains unknown; validation never assigns a license.
    """
    if manifest.sha256 is None or manifest.size_bytes is None:
        raise ValueError("headline artifacts require SHA-256 and byte size")
    source = manifest.source
    required = {
        "code",
        "runtime",
        "model",
        "workload",
        "recipe_sha256",
        "hardware",
        "environment",
        "upstream",
    }
    if required - set(source):
        raise ValueError(
            "headline provenance is missing: "
            + ", ".join(sorted(required - set(source)))
        )
    for key in ("code", "runtime", "model", "workload"):
        identity = source[key]
        if (
            not isinstance(identity, Mapping)
            or not isinstance(identity.get("identifier"), str)
            or not identity["identifier"].strip()
        ):
            raise ValueError(f"headline {key} requires an identifier")
        revision, digest = identity.get("revision"), identity.get("sha256")
        if not (
            isinstance(revision, str)
            and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision)
            or isinstance(digest, str)
            and re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise ValueError(
                f"headline {key} requires an immutable revision or SHA-256"
            )
    if not isinstance(source["recipe_sha256"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", source["recipe_sha256"]
    ):
        raise ValueError("headline recipe_sha256 must be a SHA-256 digest")
    for key in ("hardware", "environment"):
        if not isinstance(source[key], Mapping) or not source[key]:
            raise ValueError(f"headline {key} requires retained evidence")
    upstream = source["upstream"]
    if not isinstance(upstream, (list, tuple)) or not upstream:
        raise ValueError("headline upstream requires source/license metadata")
    for item in upstream:
        if not isinstance(item, Mapping) or any(
            not isinstance(item.get(key), str) or not item[key].strip()
            for key in ("source", "license", "redistribution")
        ):
            raise ValueError(
                "upstream entries require source, license, and redistribution status"
            )
    for key in ("created_at", "claim_scope", "license_status"):
        if (
            not isinstance(manifest.metadata.get(key), str)
            or not manifest.metadata[key].strip()
        ):
            raise ValueError(f"headline metadata requires {key}")
    date = datetime.fromisoformat(
        manifest.metadata["created_at"].replace("Z", "+00:00")
    )
    if date.tzinfo is None:
        raise ValueError("headline created_at requires a timezone")
    artifact_to_data(manifest)
