"""Evidence-backed model geometry and capability inspection for Metria."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ._freeze import freeze_mapping
from .identity import Capability, SupportLevel

_GEOMETRY_ALIASES: Mapping[str, tuple[str, ...]] = {
    "hidden_size": ("hidden_size", "n_embd"),
    "num_attention_heads": ("num_attention_heads", "n_head"),
    "num_key_value_heads": ("num_key_value_heads", "n_head_kv"),
    "head_dim": ("head_dim",),
    "num_hidden_layers": ("num_hidden_layers", "n_layer"),
    "context_length": ("context_length", "max_position_embeddings", "n_ctx"),
}
_VALIDATED_TURBO_HEAD_DIMS = frozenset({128, 256})


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class ModelGeometry:
    """Consistent model-geometry evidence used by capability rules.

    The type stores observations; it does not infer model-family properties from
    names. ``head_dim`` may be derived only from a self-consistent
    ``hidden_size / num_attention_heads`` pair.
    """

    hidden_size: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    num_hidden_layers: int | None = None
    context_length: int | None = None
    attention_layout: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "num_hidden_layers",
            "context_length",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _positive_int(value, name=name))
        if self.attention_layout is not None:
            object.__setattr__(
                self,
                "attention_layout",
                _text(self.attention_layout, name="attention_layout"),
            )
        if (
            self.hidden_size is not None
            and self.num_attention_heads is not None
            and self.hidden_size % self.num_attention_heads != 0
        ):
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads to derive head_dim"
            )
        derived = (
            self.hidden_size // self.num_attention_heads
            if self.hidden_size is not None and self.num_attention_heads is not None
            else None
        )
        if (
            self.head_dim is not None
            and derived is not None
            and self.head_dim != derived
        ):
            raise ValueError(
                "head_dim contradicts hidden_size / num_attention_heads "
                f"({self.head_dim} != {derived})"
            )
        if self.head_dim is None and derived is not None:
            object.__setattr__(self, "head_dim", derived)
        if (
            self.num_key_value_heads is not None
            and self.num_attention_heads is not None
            and self.num_key_value_heads > self.num_attention_heads
        ):
            raise ValueError("num_key_value_heads cannot exceed num_attention_heads")
        object.__setattr__(self, "evidence", freeze_mapping(self.evidence))

    def to_mapping(self) -> Mapping[str, Any]:
        """Return immutable normalized geometry evidence."""

        result: dict[str, Any] = {}
        for name in (
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "num_hidden_layers",
            "context_length",
            "attention_layout",
        ):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        if self.evidence:
            result["evidence"] = self.evidence
        return freeze_mapping(result)


@dataclass(frozen=True)
class GeometryInspection:
    """Normalized geometry plus the evidence-backed inspection conclusion."""

    geometry: ModelGeometry | None
    capability: Capability


def _read_alias(
    raw: Mapping[str, Any],
    canonical: str,
) -> tuple[Any | None, tuple[str, ...]]:
    aliases = _GEOMETRY_ALIASES[canonical]
    observed = [(name, raw[name]) for name in aliases if name in raw]
    if not observed:
        return None, ()
    values = {repr(value) for _, value in observed}
    if len(values) > 1:
        names = ", ".join(name for name, _ in observed)
        raise ValueError(f"conflicting {canonical} aliases: {names}")
    return observed[0][1], tuple(name for name, _ in observed)


def inspect_model_geometry(model: Mapping[str, Any]) -> GeometryInspection:
    """Normalize explicit model geometry without guessing from model-family names."""

    raw = model.get("geometry")
    if raw is None:
        return GeometryInspection(
            geometry=None,
            capability=Capability(
                name="model.geometry",
                status=SupportLevel.UNKNOWN,
                reasons=("model.geometry was not provided",),
                evidence={"source": "requested_model_metadata"},
            ),
        )
    if not isinstance(raw, Mapping):
        return GeometryInspection(
            geometry=None,
            capability=Capability(
                name="model.geometry",
                status=SupportLevel.UNKNOWN,
                reasons=("model.geometry must be a mapping",),
                evidence={"source": "requested_model_metadata"},
            ),
        )

    try:
        values: dict[str, Any] = {}
        source_keys: dict[str, tuple[str, ...]] = {}
        for canonical in _GEOMETRY_ALIASES:
            value, keys = _read_alias(raw, canonical)
            if value is not None:
                values[canonical] = value
                source_keys[canonical] = keys
        if "attention_layout" in raw:
            values["attention_layout"] = raw["attention_layout"]
            source_keys["attention_layout"] = ("attention_layout",)
        geometry = ModelGeometry(
            **values,
            evidence={
                "source": "requested_model_metadata",
                "source_keys": source_keys,
            },
        )
    except (TypeError, ValueError) as exc:
        return GeometryInspection(
            geometry=None,
            capability=Capability(
                name="model.geometry",
                status=SupportLevel.UNKNOWN,
                reasons=(str(exc),),
                evidence={
                    "source": "requested_model_metadata",
                    "provided_keys": tuple(sorted(str(key) for key in raw)),
                },
            ),
        )

    normalized = geometry.to_mapping()
    if not normalized or set(normalized) == {"evidence"}:
        return GeometryInspection(
            geometry=None,
            capability=Capability(
                name="model.geometry",
                status=SupportLevel.UNKNOWN,
                reasons=("model.geometry contains no recognized geometry fields",),
                evidence={
                    "source": "requested_model_metadata",
                    "provided_keys": tuple(sorted(str(key) for key in raw)),
                },
            ),
        )

    return GeometryInspection(
        geometry=geometry,
        capability=Capability(
            name="model.geometry",
            status=SupportLevel.SUPPORTED,
            reasons=("geometry metadata is internally consistent",),
            evidence=normalized,
        ),
    )


def _uses_turboquant(kv_config: Mapping[str, Any]) -> bool:
    for key in ("key_dtype", "value_dtype"):
        value = kv_config.get(key)
        if isinstance(value, str) and value.lower().startswith("turbo"):
            return True
    return False


def evaluate_turboquant_kv_capability(
    model: Mapping[str, Any],
    kv_config: Mapping[str, Any],
    *,
    experimental_override: bool = False,
) -> Capability:
    """Evaluate the documented TurboQuant KV-cache head-dimension guardrail."""

    if not _uses_turboquant(kv_config):
        return Capability(
            name="turboquant.kv_cache.geometry",
            status=SupportLevel.SUPPORTED,
            reasons=("requested KV cache does not use a TurboQuant dtype",),
            evidence={"active": False},
        )

    inspection = inspect_model_geometry(model)
    if inspection.geometry is None or inspection.geometry.head_dim is None:
        reasons = inspection.capability.reasons
        if inspection.geometry is not None:
            reasons = ("head_dim is unavailable from consistent geometry metadata",)
        return Capability(
            name="turboquant.kv_cache.geometry",
            status=SupportLevel.UNKNOWN,
            reasons=reasons,
            evidence={
                "active": True,
                "experimental_override": experimental_override,
                "geometry": (
                    inspection.geometry.to_mapping()
                    if inspection.geometry is not None
                    else None
                ),
                "geometry_status": SupportLevel(inspection.capability.status).value,
            },
        )

    head_dim = inspection.geometry.head_dim
    evidence = {
        "active": True,
        "head_dim": head_dim,
        "validated_head_dims": tuple(sorted(_VALIDATED_TURBO_HEAD_DIMS)),
        "experimental_override": experimental_override,
        "geometry": inspection.geometry.to_mapping(),
    }
    if head_dim <= 64:
        if experimental_override:
            return Capability(
                name="turboquant.kv_cache.geometry",
                status=SupportLevel.EXPERIMENTAL,
                reasons=(
                    "head_dim <= 64 is at/below the documented TurboQuant boundary; "
                    "explicit experimental override requested",
                ),
                evidence=evidence,
            )
        return Capability(
            name="turboquant.kv_cache.geometry",
            status=SupportLevel.UNSUPPORTED,
            reasons=(
                "head_dim <= 64 is at/below the documented TurboQuant boundary; "
                "use a non-Turbo KV dtype or opt into an explicit experimental override",
            ),
            evidence=evidence,
        )
    if head_dim in _VALIDATED_TURBO_HEAD_DIMS:
        return Capability(
            name="turboquant.kv_cache.geometry",
            status=SupportLevel.SUPPORTED,
            reasons=(f"head_dim={head_dim} is in the documented validated set",),
            evidence=evidence,
        )
    return Capability(
        name="turboquant.kv_cache.geometry",
        status=SupportLevel.EXPERIMENTAL,
        reasons=(
            f"head_dim={head_dim} is internally consistent but outside the documented "
            "validated set {128, 256}",
        ),
        evidence=evidence,
    )
