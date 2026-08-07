"""Typed identity and evidence primitives for Metria studies.

These types are intentionally mapping-compatible. They provide validated,
immutable constructors at public boundaries while normalizing into the same
mapping representation already consumed by ``RunSpec``, runtime adapters, and
``metria.study_recipe.v1``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ._freeze import freeze_mapping

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _optional_text(value: str | None, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string when provided")
    return value


def _required_text(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _extension_mapping(
    value: Mapping[str, Any],
    *,
    name: str,
    reserved: frozenset[str],
) -> Mapping[str, Any]:
    collisions = sorted(set(value) & reserved)
    if collisions:
        raise ValueError(
            f"{name} cannot redefine reserved fields: {', '.join(collisions)}"
        )
    return freeze_mapping(value)


class _MappingPrimitive(Mapping[str, Any]):
    """A validated public object that normalizes to Metria's mapping boundary."""

    def to_mapping(self) -> Mapping[str, Any]:
        """Return the immutable mapping representation retained by Metria."""

        raise NotImplementedError

    def __getitem__(self, key: str) -> Any:
        return self.to_mapping()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_mapping())

    def __len__(self) -> int:
        return len(self.to_mapping())


@dataclass(frozen=True)
class ModelRef(_MappingPrimitive):
    """Requested model/tokenizer identity without pretending it is observed truth."""

    id: str | None = None
    path: str | os.PathLike[str] | None = None
    revision: str | None = None
    tokenizer_id: str | None = None
    tokenizer_revision: str | None = None
    geometry: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        model_id = _optional_text(self.id, name="model id")
        path = self.path
        if path is not None:
            path = os.fspath(path)
            if not isinstance(path, str) or not path.strip():
                raise ValueError("model path must be a non-empty path when provided")
        if model_id is None and path is None:
            raise ValueError("ModelRef requires at least an id or local path")
        object.__setattr__(self, "id", model_id)
        object.__setattr__(self, "path", path)
        object.__setattr__(
            self,
            "revision",
            _optional_text(self.revision, name="model revision"),
        )
        object.__setattr__(
            self,
            "tokenizer_id",
            _optional_text(self.tokenizer_id, name="tokenizer id"),
        )
        object.__setattr__(
            self,
            "tokenizer_revision",
            _optional_text(self.tokenizer_revision, name="tokenizer revision"),
        )
        object.__setattr__(self, "geometry", freeze_mapping(self.geometry))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def to_mapping(self) -> Mapping[str, Any]:
        result: dict[str, Any] = {}
        if self.id is not None:
            result["id"] = self.id
        if self.path is not None:
            result["path"] = self.path
        if self.revision is not None:
            result["revision"] = self.revision
        if self.tokenizer_id is not None:
            result["tokenizer_id"] = self.tokenizer_id
        if self.tokenizer_revision is not None:
            result["tokenizer_revision"] = self.tokenizer_revision
        if self.geometry:
            result["geometry"] = self.geometry
        if self.metadata:
            result["metadata"] = self.metadata
        return freeze_mapping(result)


@dataclass(frozen=True)
class RuntimeConfig(_MappingPrimitive):
    """Requested runtime identity plus runtime-specific extension fields."""

    name: str
    version: str | None = None
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _required_text(self.name, name="runtime name"))
        object.__setattr__(
            self,
            "version",
            _optional_text(self.version, name="runtime version"),
        )
        object.__setattr__(
            self,
            "config",
            _extension_mapping(
                self.config,
                name="runtime config",
                reserved=frozenset({"name", "version"}),
            ),
        )

    def to_mapping(self) -> Mapping[str, Any]:
        result: dict[str, Any] = {"name": self.name}
        if self.version is not None:
            result["version"] = self.version
        result.update(self.config)
        return freeze_mapping(result)


@dataclass(frozen=True)
class WorkloadSpec(_MappingPrimitive):
    """Requested scenario/workload configuration retained by ``RunSpec``."""

    name: str | None = None
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = _optional_text(self.name, name="workload name")
        config = _extension_mapping(
            self.config,
            name="workload config",
            reserved=frozenset({"name"}),
        )
        if name is None and not config:
            raise ValueError("WorkloadSpec requires a name or configuration")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "config", config)

    def to_mapping(self) -> Mapping[str, Any]:
        result: dict[str, Any] = {}
        if self.name is not None:
            result["name"] = self.name
        result.update(self.config)
        return freeze_mapping(result)


class SupportLevel(str, Enum):
    """Conservative capability/support states shared across Metria."""

    SUPPORTED = "supported"
    EXPERIMENTAL = "experimental"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Capability:
    """One named capability conclusion with reasons and retained evidence."""

    name: str
    status: SupportLevel | str
    reasons: tuple[str, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name", _required_text(self.name, name="capability name")
        )
        try:
            status = SupportLevel(self.status)
        except ValueError as exc:
            supported = ", ".join(level.value for level in SupportLevel)
            raise ValueError(f"capability status must be one of: {supported}") from exc
        reasons = tuple(self.reasons)
        if any(not isinstance(reason, str) or not reason.strip() for reason in reasons):
            raise ValueError("capability reasons must be non-empty strings")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "evidence", freeze_mapping(self.evidence))

    def to_mapping(self) -> Mapping[str, Any]:
        return freeze_mapping(
            {
                "status": SupportLevel(self.status).value,
                "reasons": self.reasons,
                "evidence": self.evidence,
            }
        )


@dataclass(frozen=True)
class CapabilitySet:
    """Immutable set of uniquely named capability conclusions."""

    capabilities: tuple[Capability, ...] = ()

    def __post_init__(self) -> None:
        capabilities = tuple(self.capabilities)
        names = [capability.name for capability in capabilities]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                "capability names must be unique: " + ", ".join(duplicates)
            )
        object.__setattr__(self, "capabilities", capabilities)

    def get(self, name: str) -> Capability | None:
        """Return a capability by name, or ``None`` when it was not evaluated."""

        for capability in self.capabilities:
            if capability.name == name:
                return capability
        return None

    def to_mapping(self) -> Mapping[str, Any]:
        return freeze_mapping(
            {
                capability.name: capability.to_mapping()
                for capability in self.capabilities
            }
        )


@dataclass(frozen=True)
class HardwareFingerprint(_MappingPrimitive):
    """Observed hardware/software identity, separate from requested placement."""

    platform: Mapping[str, Any] = field(default_factory=dict)
    host: Mapping[str, Any] = field(default_factory=dict)
    accelerators: tuple[Mapping[str, Any], ...] = ()
    software: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "platform", freeze_mapping(self.platform))
        object.__setattr__(self, "host", freeze_mapping(self.host))
        object.__setattr__(
            self,
            "accelerators",
            tuple(freeze_mapping(accelerator) for accelerator in self.accelerators),
        )
        object.__setattr__(self, "software", freeze_mapping(self.software))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def to_mapping(self) -> Mapping[str, Any]:
        return freeze_mapping(
            {
                "platform": self.platform,
                "host": self.host,
                "accelerators": self.accelerators,
                "software": self.software,
                "metadata": self.metadata,
            }
        )


@dataclass(frozen=True)
class ArtifactManifest(_MappingPrimitive):
    """Identity and provenance for one external or generated artifact."""

    name: str
    kind: str
    uri: str | None = None
    path: str | os.PathLike[str] | None = None
    revision: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    source: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name", _required_text(self.name, name="artifact name")
        )
        object.__setattr__(
            self, "kind", _required_text(self.kind, name="artifact kind")
        )
        object.__setattr__(self, "uri", _optional_text(self.uri, name="artifact uri"))
        path = self.path
        if path is not None:
            path = os.fspath(path)
            if not isinstance(path, str) or not path.strip():
                raise ValueError("artifact path must be a non-empty path when provided")
        object.__setattr__(self, "path", path)
        object.__setattr__(
            self,
            "revision",
            _optional_text(self.revision, name="artifact revision"),
        )
        sha256 = self.sha256
        if sha256 is not None:
            if not isinstance(sha256, str) or _SHA256_RE.fullmatch(sha256) is None:
                raise ValueError(
                    "artifact sha256 must be exactly 64 hexadecimal characters"
                )
            sha256 = sha256.lower()
        object.__setattr__(self, "sha256", sha256)
        if self.size_bytes is not None:
            if isinstance(self.size_bytes, bool) or not isinstance(
                self.size_bytes, int
            ):
                raise TypeError("artifact size_bytes must be an integer when provided")
            if self.size_bytes < 0:
                raise ValueError("artifact size_bytes must be non-negative")
        if self.uri is None and self.path is None and self.sha256 is None:
            raise ValueError(
                "ArtifactManifest requires a uri, path, or sha256 identity"
            )
        object.__setattr__(self, "source", freeze_mapping(self.source))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def to_mapping(self) -> Mapping[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
        }
        if self.uri is not None:
            result["uri"] = self.uri
        if self.path is not None:
            result["path"] = self.path
        if self.revision is not None:
            result["revision"] = self.revision
        if self.sha256 is not None:
            result["sha256"] = self.sha256
        if self.size_bytes is not None:
            result["size_bytes"] = self.size_bytes
        if self.source:
            result["source"] = self.source
        if self.metadata:
            result["metadata"] = self.metadata
        return freeze_mapping(result)
