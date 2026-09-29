"""Prepare a bounded local serving concurrency experiment from immutable inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, TextIO

from .measurements.serving_impact import ServingImpactAnalysis
from .measurements.trajectory import TokenTrajectoryProtocol
from .preparation import prepare_vllm_prefix_recipe
from .recipes import StudyRecipe, study_recipe_digest, study_recipe_to_json
from .verification_serving import (
    SERVING_CONTROLS,
    SERVING_VARIATIONS,
    trial_identity,
    validate_serving_recipe,
)


def prepare_serving_recipe(
    model_dir: str | Path,
    descriptor: Mapping[str, Any],
    prompts: Sequence[Mapping[str, Any]],
    **options: Any,
) -> StudyRecipe:
    original = prepare_vllm_prefix_recipe(model_dir, descriptor, prompts, **options)
    config = original.measurement_configs[TokenTrajectoryProtocol.name]
    runs = tuple(
        replace(
            run,
            runtime={**run.runtime, "max_num_seqs": 2, "enable_prefix_caching": False},
            trial_policy=trial_identity(config, concurrency),
        )
        for run, concurrency in zip(original.study.runs, (1, 2), strict=True)
    )
    comparison = replace(
        original.study.comparison,
        vary=SERVING_VARIATIONS,
        control=SERVING_CONTROLS,
        analyses=(*original.study.comparison.analyses, ServingImpactAnalysis.name),
    )
    recipe = replace(
        original,
        study=replace(
            original.study,
            name="vllm-local-serving-concurrency",
            runs=runs,
            comparison=comparison,
        ),
    )
    validate_serving_recipe(recipe)
    return recipe


def add_serving_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "prepare-vllm-serving",
        help="prepare local native streaming and concurrency measurements",
    )
    for name in ("model", "descriptor", "workload", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--warmup-trials", type=int, default=1)
    parser.add_argument("--measured-trials", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900)


def prepare_serving_command(args: Any, stdout: TextIO) -> int:
    from .onboarding import _json, _read, _runtime_host_supported
    from .policies import policy_from_data
    from .verification import _write_atomic

    if args.output.exists():
        raise FileExistsError("recipe output already exists; choose a new file")
    if not _runtime_host_supported():
        raise ValueError(
            "local vLLM serving preparation requires Linux (including WSL)"
        )
    descriptor = _json(_read(args.descriptor))
    if not isinstance(descriptor, Mapping):
        raise ValueError("model descriptor must contain trusted file pins")
    prompts = [
        _json(line) for line in _read(args.workload).splitlines() if line.strip()
    ]
    recipe = prepare_serving_recipe(
        args.model,
        descriptor,
        prompts,
        context=args.context,
        max_tokens=args.max_tokens,
        warmup_trials=args.warmup_trials,
        measured_trials=args.measured_trials,
        timeout_s=args.timeout,
    )
    if args.policy is not None:
        recipe = replace(recipe, policy=policy_from_data(_json(_read(args.policy))))
    _write_atomic(args.output, study_recipe_to_json(recipe) + "\n")
    stdout.write(f"Prepared {args.output}\nRecipe: {study_recipe_digest(recipe)}\n")
    return 0
