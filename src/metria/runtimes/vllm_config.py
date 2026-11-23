"""vLLM model-source, KV-cache treatment and applied-config helpers.

Split from :mod:`metria.runtimes.vllm` to keep that module within its size
ceiling; ``vllm`` re-exports these names.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..models import RunSpec, TreatmentSpec, TreatmentType
from .vllm_treatments import is_staged_treatment

_SUPPORTED_KV_DTYPES = frozenset({"auto", "fp8", "fp8_e4m3", "fp8_e5m2"})
_SUPPORTED_KV_KEYS = frozenset({"dtype"})


def _model_source(spec: RunSpec) -> dict[str, Any]:
    """Resolve a local model path or registry identifier for vLLM."""

    raw_path = spec.model.get("path")
    raw_id = spec.model.get("id")
    if raw_path is not None:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise TypeError("model.path must be a non-empty string")
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"model.path does not exist: {path}")
        model = str(path)
        source_kind = "local"
    elif raw_id is not None:
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise TypeError("model.id must be a non-empty string")
        model = raw_id
        source_kind = "registry"
    else:
        raise ValueError("vLLM requires model.id or model.path")

    revision = spec.model.get("revision")
    raw_tokenizer_id = spec.model.get("tokenizer_id")
    tokenizer_revision = spec.model.get("tokenizer_revision")
    if tokenizer_revision is None and raw_tokenizer_id is None:
        tokenizer_revision = revision
    if revision is not None and not isinstance(revision, str):
        raise TypeError("model.revision must be a string or null")
    if raw_tokenizer_id is not None and (
        not isinstance(raw_tokenizer_id, str) or not raw_tokenizer_id.strip()
    ):
        raise TypeError("model.tokenizer_id must be a non-empty string or null")
    if tokenizer_revision is not None and not isinstance(tokenizer_revision, str):
        raise TypeError("model.tokenizer_revision must be a string or null")
    tokenizer = raw_tokenizer_id or model
    return {
        "model": model,
        "source_kind": source_kind,
        "requested_id": raw_id,
        "requested_path": str(raw_path) if raw_path is not None else None,
        "revision": revision,
        "tokenizer": tokenizer,
        "requested_tokenizer_id": raw_tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
    }


def _kv_treatment(treatments: Sequence[TreatmentSpec]) -> dict[str, Any]:
    """Resolve the vLLM-native KV-cache treatment for one run."""

    dtype = "auto"
    seen = False
    for treatment in treatments:
        if is_staged_treatment(treatment):
            continue  # validated by resolve_staged_treatments
        if treatment.name not in {"kv_cache", "vllm.kv_cache"}:
            raise ValueError(
                "vLLM adapter does not yet apply treatment "
                f"{treatment.name!r}; materialize or adapt it before runtime launch"
            )
        if treatment.kind is not TreatmentType.RUNTIME_FEATURE:
            raise ValueError("vLLM KV-cache treatment must use kind='runtime_feature'")
        if seen:
            raise ValueError("only one vLLM KV-cache treatment may be applied")
        unknown = sorted(set(treatment.config) - _SUPPORTED_KV_KEYS)
        if unknown:
            raise ValueError(
                "unsupported vLLM KV-cache treatment keys: " + ", ".join(unknown)
            )
        requested = treatment.config.get("dtype", "auto")
        if not isinstance(requested, str):
            raise TypeError("vLLM KV-cache dtype must be a string")
        if requested not in _SUPPORTED_KV_DTYPES:
            raise ValueError(
                f"unsupported vLLM KV-cache dtype {requested!r}; "
                f"supported values: {', '.join(sorted(_SUPPORTED_KV_DTYPES))}"
            )
        dtype = requested
        seen = True
    return {"dtype": dtype}


def _evidence_scalar(value: Any) -> Any:
    """Normalize introspected runtime fields into immutable evidence scalars."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _introspect_applied_config(llm: Any) -> dict[str, Any]:
    """Best-effort introspection of fields vLLM exposes after engine creation."""

    fields: dict[str, Any] = {}
    model_config = getattr(llm, "model_config", None)
    if model_config is not None:
        for attr in (
            "model",
            "dtype",
            "max_model_len",
            "revision",
            "tokenizer",
            "tokenizer_revision",
        ):
            if hasattr(model_config, attr):
                fields[f"model.{attr}"] = _evidence_scalar(getattr(model_config, attr))

    engine = getattr(llm, "llm_engine", None)
    vllm_config = getattr(engine, "vllm_config", None)
    cache_config = getattr(vllm_config, "cache_config", None)
    if cache_config is None:
        cache_config = getattr(engine, "cache_config", None)
    if cache_config is not None:
        for attr in ("cache_dtype", "gpu_memory_utilization", "enable_prefix_caching"):
            if hasattr(cache_config, attr):
                fields[f"cache.{attr}"] = _evidence_scalar(getattr(cache_config, attr))

    parallel_config = getattr(vllm_config, "parallel_config", None)
    for attr in ("tensor_parallel_size", "pipeline_parallel_size"):
        if parallel_config is not None and hasattr(parallel_config, attr):
            fields[f"parallel.{attr}"] = _evidence_scalar(
                getattr(parallel_config, attr)
            )

    return {
        "status": "introspected" if fields else "unverified",
        "fields": fields,
    }
