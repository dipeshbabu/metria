"""Claimed identity for remote endpoints, with consistency instead of authority.

A client cannot hash a remote server's weights or binary. Everything a server
reports about itself is therefore recorded with authority ``partial`` and
source ``server_reported``, so aggregate identity can never become
``verified``. What Metria *can* enforce is consistency:

* the claims snapshot taken at launch must equal the one taken after the last
  request (a rolling deploy or a different replica shows up as drift);
* a claim that contradicts the operator's ``declared`` value fails the run;
* the server's own tokenizer must agree with the locally pinned tokenizer on
  every prompt it was asked about.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from ..identity_evidence import (
    IdentityStatus,
    RuntimeIdentityEvidence,
    aggregate_identity_status,
)
from .endpoint_http import EndpointConnection, EndpointError

DECLARED_KEYS = frozenset(
    {
        "engine",
        "engine_version",
        "model_root",
        "model_revision",
        "max_model_len",
        "dtype",
        "kv_cache_dtype",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "hardware",
    }
)
# declared key -> (claims component, claimed field)
_CHECKED = {
    "engine_version": ("runtime", "version"),
    "max_model_len": ("model", "max_model_len"),
    "model_root": ("model", "root"),
}
SNAPSHOT_GAP = "server-reported identity changed during the run"
TOKENIZER_GAP = "server tokenizer disagrees with the pinned local tokenizer"


def validate_declared(value: Any) -> dict[str, Any]:
    """Accept only allowlisted scalar operator statements."""

    if value is None:
        return {}
    if not isinstance(value, Mapping) or set(value) - DECLARED_KEYS:
        raise ValueError(
            "runtime.declared accepts only: " + ", ".join(sorted(DECLARED_KEYS))
        )
    for key, item in value.items():
        if isinstance(item, bool) or not isinstance(item, (str, int)):
            raise ValueError(f"runtime.declared.{key} must be a string or integer")
        if isinstance(item, str) and not 0 < len(item) <= 256:
            raise ValueError(f"runtime.declared.{key} must be 1-256 characters")
    return dict(value)


def _optional(connection: EndpointConnection, path: str) -> Any:
    try:
        payload, _ = connection.request("GET", path)
    except EndpointError:
        return None
    return payload


def collect_claims(
    connection: EndpointConnection, flavor: str, model_id: str
) -> dict[str, dict[str, Any]]:
    """Read what the server says about itself; absent facts stay absent."""

    payload, _ = connection.request("GET", "/v1/models")
    rows = payload.get("data") if isinstance(payload, Mapping) else None
    served = [
        row
        for row in (rows if isinstance(rows, list) else [])
        if isinstance(row, Mapping) and row.get("id") == model_id
    ]
    if len(served) != 1:
        raise RuntimeError("the endpoint does not serve the requested model id")
    model = {
        key: served[0][key]
        for key in ("id", "root", "max_model_len")
        if isinstance(served[0].get(key), (str, int))
        and not isinstance(served[0].get(key), bool)
    }
    root = model.get("root")
    if isinstance(root, str) and _is_path(root):
        # Servers often report a private filesystem path; retain only its hash.
        del model["root"]
        model["root_sha256"] = _sha256(root)
    runtime: dict[str, Any] = {"flavor": flavor}
    if flavor == "vllm":
        version = _optional(connection, "/version")
        if isinstance(version, Mapping) and isinstance(version.get("version"), str):
            runtime["version"] = version["version"]
    return {"model": model, "runtime": runtime}


def _is_path(value: str) -> bool:
    return value.startswith(("/", "~", ".")) or "\\" in value or ":" in value


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def check_declared(
    declared: Mapping[str, Any], claims: Mapping[str, Mapping[str, Any]]
) -> None:
    """Fail before measurement when the server contradicts the operator."""

    root = declared.get("model_root")
    claimed_hash = claims["model"].get("root_sha256")
    if isinstance(root, str) and claimed_hash and _sha256(root) != claimed_hash:
        raise RuntimeError(
            "server-reported model.root contradicts runtime.declared.model_root"
        )
    for key, (component, field) in _CHECKED.items():
        if key in declared and field in claims[component]:
            if claims[component][field] != declared[key]:
                raise RuntimeError(
                    f"server-reported {component}.{field} contradicts runtime.declared.{key}"
                )


class TokenizerCheck:
    """Compare server ``/tokenize`` with the pinned local tokenizer per prompt."""

    def __init__(self, connection: EndpointConnection, model_id: str) -> None:
        self._connection = connection
        self._model_id = model_id
        self._seen: set[str] = set()
        self.checked = 0
        self.mismatched = 0
        self.available = True

    def check(self, text: str, local: Sequence[int], *, special: bool = True) -> None:
        if not self.available or text in self._seen:
            return
        self._seen.add(text)
        try:
            payload, _ = self._connection.request(
                "POST",
                "/tokenize",
                {
                    "model": self._model_id,
                    "prompt": text,
                    "add_special_tokens": special,
                },
            )
        except EndpointError:
            self.available = False
            return
        tokens = payload.get("tokens") if isinstance(payload, Mapping) else None
        self.checked += 1
        if tokens != list(local):
            self.mismatched += 1

    def evidence(self) -> dict[str, Any]:
        if not self.available and not self.checked:
            status = IdentityStatus.UNKNOWN
        elif self.mismatched:
            status = IdentityStatus.MISMATCH
        else:
            status = IdentityStatus.PARTIAL
        return {
            "status": status.value,
            "source": "pinned_local_vs_server_tokenize",
            "checked_prompts": self.checked,
            "mismatched_prompts": self.mismatched,
        }


def identity_evidence(
    claims: Mapping[str, Mapping[str, Any]],
    endpoint: Mapping[str, Any],
    tokenizer: Mapping[str, Any] | None,
    chat_template: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build ``RuntimeIdentityEvidence`` with every server claim as partial."""

    model = (
        {"status": "partial", "source": "server_reported", **claims["model"]}
        if claims["model"]
        else {}
    )
    runtime = {"status": "partial", "source": "server_reported", **claims["runtime"]}
    components = [model, runtime, tokenizer or {}, chat_template or {}]
    status = aggregate_identity_status([row["status"] for row in components if row])
    evidence = RuntimeIdentityEvidence(
        status=status,
        model=model,
        runtime=runtime,
        tokenizer=tokenizer or {},
        chat_template=chat_template or {},
        endpoint=endpoint,
        reasons=("server-reported identity is claimed, not verified by Metria",),
    )
    return dict(evidence.to_mapping())


def identity_gaps(observed: Mapping[str, Any]) -> list[str]:
    """Evidence gaps a verification profile must apply to an endpoint run."""

    gaps = []
    snapshots = observed.get("identity_snapshots")
    if not isinstance(snapshots, Mapping) or snapshots.get("stable") is not True:
        gaps.append(SNAPSHOT_GAP)
    identity = observed.get("identity")
    tokenizer = identity.get("tokenizer") if isinstance(identity, Mapping) else None
    if isinstance(tokenizer, Mapping) and tokenizer.get("status") == "mismatch":
        gaps.append(TOKENIZER_GAP)
    return gaps
