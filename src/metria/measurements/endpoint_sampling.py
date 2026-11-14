"""Production decode settings for endpoint workloads.

Greedy runs keep one ``seed``. Sampled runs (``temperature > 0``) declare a
list of ``seeds``: every prompt is expanded into one row per seed, each with
its own per-request seed, so reference and candidate draw the same seeded
samples and pass rates are estimated over all of them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

SCENARIO_KEYS = frozenset(
    {"max_tokens", "temperature", "seed", "seeds", "top_p", "stop", "chat_template"}
)


def _number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _seed(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and 0 <= value < 2**31


def _validate_seeds(scenario: Mapping[str, Any], temperature: float) -> None:
    if temperature == 0:
        if not _seed(scenario.get("seed")) or "seeds" in scenario:
            raise ValueError(
                "greedy endpoint runs need one integer seed and no seeds list"
            )
        return
    seeds = scenario.get("seeds")
    if (
        "seed" in scenario
        or not isinstance(seeds, (list, tuple))
        or not 1 <= len(seeds) <= 32
        or not all(_seed(item) for item in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError(
            "sampled endpoint runs (temperature > 0) need 1-32 distinct integer "
            "seeds and no single seed"
        )


def validate_scenario(scenario: Mapping[str, Any]) -> None:
    """Accept greedy or seeded sampling with optional top_p, stop and chat."""

    if unknown := set(scenario) - SCENARIO_KEYS:
        raise ValueError(
            "unsupported endpoint scenario keys: " + ", ".join(sorted(unknown))
        )
    temperature: Any = scenario.get("temperature")
    if not _number(temperature) or not 0 <= temperature <= 2:
        raise ValueError("scenario.temperature must be a number from 0 to 2")
    _validate_seeds(scenario, temperature)
    top_p = scenario.get("top_p", 1.0)
    if not _number(top_p) or not 0 < top_p <= 1:
        raise ValueError("scenario.top_p must be in (0, 1]")
    stop = scenario.get("stop", [])
    if (
        not isinstance(stop, (list, tuple))
        or len(stop) > 4
        or not all(isinstance(item, str) and 0 < len(item) <= 32 for item in stop)
    ):
        raise ValueError("scenario.stop must be at most 4 strings of 1-32 characters")
    if not isinstance(scenario.get("chat_template", False), bool):
        raise ValueError("scenario.chat_template must be a boolean")


def expand_seeds(
    config: Mapping[str, Any], scenario: Mapping[str, Any]
) -> dict[str, Any]:
    """Expand prompts into one row per declared seed for sampled runs."""

    seeds = scenario.get("seeds")
    if not seeds:
        return dict(config)
    rows = []
    for row in config["prompts"]:
        for seed in seeds:
            generation = dict(row.get("generation", {}))
            generation["seed"] = seed
            rows.append(
                {**row, "id": f"{row['id']}@seed{seed}", "generation": generation}
            )
    return {**config, "prompts": rows}
