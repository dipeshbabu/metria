"""Shared strict input validation for qualified llama.cpp CPU profiles."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .measurements import TokenTrajectoryProtocol
from .models import RunSpec
from .protocols import InferenceRequest
from .runtimes.llamacpp import (
    _generation_options,
    _requested_model_sha256,
    _runtime_config,
)


def validate_cpu_run(
    run: RunSpec, index: int, config: Mapping[str, Any], registries: Any
) -> str:
    name = run.runtime.get("name")
    if name not in registries.adapters:
        raise ValueError(f"run[{index}] runtime is not registered for verification")
    if name != "llamacpp":
        raise ValueError("the first verify workflow supports local llama.cpp only")
    if (
        run.measurements != (TokenTrajectoryProtocol.name,)
        or TokenTrajectoryProtocol.name not in registries.measurements
    ):
        raise ValueError(
            "local verify requires the registered decode-time trajectory measurement"
        )
    if run.treatments or run.trial_policy or run.environment_selector:
        raise ValueError(
            "local CPU verification does not yet apply treatments, trial policies, or environment selectors"
        )
    if set(run.runtime) - {
        "name",
        "bin_dir",
        "n_gpu_layers",
        "flash_attention",
        "threads",
        "threads_batch",
        "extra_args",
    }:
        raise ValueError("unsupported local verification runtime fields")
    if set(run.model) - {"path", "sha256", "id", "revision", "geometry"}:
        raise ValueError("unsupported local verification model fields")
    if "system" in run.scenario:
        raise ValueError("the plain-completion verifier does not accept system prompts")
    if set(run.scenario) - {
        "name",
        "context",
        "max_tokens",
        "seed",
        "temperature",
        "timeout",
        "chat_template",
        "reasoning",
    }:
        raise ValueError("unsupported local verification scenario fields")
    runtime = _runtime_config(run)
    if runtime["n_gpu_layers"] != 0 or runtime["extra_args"]:
        raise ValueError("local verify requires n_gpu_layers=0 and no extra_args")
    if runtime["threads"] is None or runtime["threads_batch"] is None:
        raise ValueError("local verify requires explicit threads and threads_batch")
    pin = _requested_model_sha256(run)
    if pin is None:
        raise ValueError(
            "local verify requires model.sha256 from a trusted artifact manifest"
        )
    _validate_generation(run, config, registries)
    return pin


def _validate_generation(
    run: RunSpec, config: Mapping[str, Any], registries: Any
) -> None:
    registries.measurements[TokenTrajectoryProtocol.name].requirements(config)
    for row in config["prompts"]:
        generation = {**config.get("generation", {}), **row.get("generation", {})}
        options = _generation_options(
            InferenceRequest(row["prompt"], generation), run.scenario
        )
        if (
            options["temperature"] != 0.0
            or options["chat_template"]
            or options["system"]
        ):
            raise ValueError(
                "local verify requires greedy plain completion: temperature=0 and chat_template=false"
            )
        if options["max_tokens"] <= 0:
            raise ValueError("local verify requires a positive generation length")
