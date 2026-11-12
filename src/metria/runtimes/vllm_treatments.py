"""STAGED vLLM runtime treatments beyond the qualified KV-cache dtype.

Status: STAGED. These treatments map to documented vLLM engine arguments and
are read back from the constructed engine, but no retained native run
qualifies them. The built-in ``vllm.staged_treatments`` capability check
classifies every use as EXPERIMENTAL, so preflight blocks it unless the run
lists that check in ``trial_policy.capability_overrides``. The qualified
``metria verify`` profiles continue to reject them.

Each treated value follows requested -> resolved -> observed: the request is
validated here, passed to ``vllm.LLM`` as an engine argument, and compared with
the engine configuration after construction. A contradicting readback fails
the launch; a field the engine does not expose is recorded as unobserved.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..models import TreatmentSpec, TreatmentType

QUALIFICATION = "staged"
CAPABILITY_CHECK = "vllm.staged_treatments"


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _block_size(value: Any, name: str) -> int:
    if isinstance(value, bool) or value not in {8, 16, 32, 64, 128}:
        raise ValueError(f"{name} must be one of 8, 16, 32, 64, or 128")
    return int(value)


# treatment name -> engine argument -> (validator, readback config, attribute)
_Field = tuple[Callable[[Any, str], Any], str, str]
STAGED_TREATMENTS: Mapping[str, Mapping[str, _Field]] = {
    "vllm.scheduler": {
        "enable_chunked_prefill": (
            _boolean,
            "scheduler_config",
            "enable_chunked_prefill",
        ),
        "max_num_batched_tokens": (
            _positive_int,
            "scheduler_config",
            "max_num_batched_tokens",
        ),
    },
    "vllm.cache_layout": {
        "block_size": (_block_size, "cache_config", "block_size"),
    },
}


def is_staged_treatment(treatment: TreatmentSpec) -> bool:
    return treatment.name in STAGED_TREATMENTS


def resolve_staged_treatments(
    treatments: Sequence[TreatmentSpec],
) -> dict[str, dict[str, Any]]:
    """Validate staged treatments and return normalized engine arguments."""

    resolved: dict[str, dict[str, Any]] = {}
    for treatment in treatments:
        fields = STAGED_TREATMENTS.get(treatment.name)
        if fields is None:
            continue
        if treatment.kind is not TreatmentType.RUNTIME_FEATURE:
            raise ValueError(f"{treatment.name} must use kind='runtime_feature'")
        if treatment.name in resolved:
            raise ValueError(f"only one {treatment.name} treatment may be applied")
        if not treatment.config:
            raise ValueError(f"{treatment.name} requires at least one setting")
        unknown = sorted(set(treatment.config) - set(fields))
        if unknown:
            raise ValueError(
                f"unsupported {treatment.name} settings: " + ", ".join(unknown)
            )
        resolved[treatment.name] = {
            key: fields[key][0](value, f"{treatment.name}.{key}")
            for key, value in sorted(treatment.config.items())
        }
    return resolved


def engine_arguments(resolved: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Flatten resolved staged treatments into ``vllm.LLM`` keyword arguments."""

    return {
        key: value for settings in resolved.values() for key, value in settings.items()
    }


def _engine_config(llm: Any, name: str) -> Any:
    engine = getattr(llm, "llm_engine", None)
    vllm_config = getattr(engine, "vllm_config", None)
    config = getattr(vllm_config, name, None)
    return config if config is not None else getattr(engine, name, None)


def read_back_staged_treatments(
    llm: Any, resolved: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Compare requested staged settings with the constructed engine config."""

    rows: dict[str, dict[str, Any]] = {}
    contradictions: list[str] = []
    for name, settings in resolved.items():
        for key, requested in settings.items():
            _, config_name, attribute = STAGED_TREATMENTS[name][key]
            config = _engine_config(llm, config_name)
            if config is None or not hasattr(config, attribute):
                rows[f"{name}.{key}"] = {"requested": requested, "status": "unobserved"}
                continue
            observed = getattr(config, attribute)
            matches = type(observed) is type(requested) and observed == requested
            rows[f"{name}.{key}"] = {
                "requested": requested,
                "observed": observed
                if isinstance(observed, (bool, int, float, str)) or observed is None
                else str(observed),
                "source": f"{config_name}.{attribute}",
                "status": "applied" if matches else "contradicted",
            }
            if not matches:
                contradictions.append(f"{name}.{key}")
    if contradictions:
        raise RuntimeError(
            "vLLM engine did not apply staged treatment settings: "
            + ", ".join(contradictions)
        )
    return {"qualification": QUALIFICATION, "settings": rows}
