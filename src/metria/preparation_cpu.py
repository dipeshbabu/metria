"""Preparation and native capture qualification for llama.cpp build comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, TextIO

from .execution import execute_run
from .measurements import TokenTrajectoryProtocol, TrajectoryAgreementAnalysis
from .measurements.verification_impact import VerificationImpactAnalysis
from .models import ComparisonPlan, RunSpec, RunStatus, StudySpec
from .policies import policy_from_data
from .recipes import StudyRecipe, study_recipe_digest, study_recipe_to_json
from .records import run_record_to_json
from .runtimes.llamacpp import LlamaCppAdapter, _find_binary, _sha256_file
from .runtimes.llamacpp_qualified import PROVIDERS_KEY
from .verification_build import BUILD_VARIATIONS, build_route


def prepare_llamacpp_build_recipe(
    reference_bin_dir: Path,
    candidate_bin_dir: Path,
    model: Path,
    model_sha256: str,
    prompts: Sequence[Mapping[str, Any]],
    *,
    threads: int = 2,
    context: int = 256,
    max_tokens: int = 16,
    timeout: float = 300,
) -> StudyRecipe:
    """Pin both provider contents and construct a strictly controlled recipe."""
    directories = [
        path.expanduser().resolve() for path in (reference_bin_dir, candidate_bin_dir)
    ]
    providers = {}
    for directory in directories:
        binary = _find_binary(directory, "llama-completion")
        if binary is None:
            raise ValueError(
                "each build needs its own patched llama-completion provider"
            )
        providers[str(directory)] = _sha256_file(binary)
    runs = tuple(
        RunSpec(
            model={"path": str(model.expanduser().resolve()), "sha256": model_sha256},
            runtime={
                "name": "llamacpp",
                "bin_dir": str(directory),
                "threads": threads,
                "threads_batch": 1,
                "n_gpu_layers": 0,
                "flash_attention": False,
            },
            scenario={
                "context": context,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "seed": 42,
                "chat_template": False,
                "timeout": timeout,
            },
            measurements=(TokenTrajectoryProtocol.name,),
        )
        for directory in directories
    )
    recipe = StudyRecipe(
        study=StudySpec(
            name="llamacpp-cpu-build-change",
            runs=runs,
            comparison=ComparisonPlan(
                vary=BUILD_VARIATIONS,
                control=frozenset({"model", "scenario", "measurements"}),
                analyses=(
                    TrajectoryAgreementAnalysis.name,
                    VerificationImpactAnalysis.name,
                ),
            ),
        ),
        measurement_configs={TokenTrajectoryProtocol.name: {"prompts": list(prompts)}},
        environment={PROVIDERS_KEY: providers},
    )
    build_route(recipe)
    return recipe


def qualify_cpu_providers(
    recipe: StudyRecipe,
    paths: Sequence[Path],
    *,
    providers: Mapping[str, str] | None = None,
) -> None:
    """Retain each real qualification immediately, including failed attempts."""
    from .verification import _evidence_gaps, _write_atomic

    pins = recipe.environment[PROVIDERS_KEY] if providers is None else providers
    for role, run, output in zip(
        ("reference", "candidate"), recipe.study.runs, paths, strict=True
    ):
        pin = pins[run.runtime["bin_dir"]]
        record = execute_run(
            study_name="llamacpp-capture-provider-qualification",
            run_id=role,
            spec=run,
            adapter=LlamaCppAdapter(),
            measurement=TokenTrajectoryProtocol(),
            measurement_config={
                "prompts": [{"id": "capture-probe", "prompt": "Once upon a time"}],
                "generation": {"max_tokens": 1},
            },
            environment={"llama_cpp_token_ids_capture_sha256": pin},
        )
        _write_atomic(output, run_record_to_json(record) + "\n")
        if record.status is not RunStatus.COMPLETED or _evidence_gaps(record, pin):
            raise ValueError(f"{role} provider qualification failed; inspect {output}")


def prepare_build_command(args: Any, stdout: TextIO) -> int:
    from .onboarding import _json, _read
    from .verification import _write_atomic

    output = args.output.expanduser().resolve()
    qualifications = [
        output.with_suffix(f".{role}.qualification.run.json")
        for role in ("reference", "candidate")
    ]
    if any(path.exists() for path in (output, *qualifications)):
        raise FileExistsError(
            "choose new recipe and qualification paths; existing evidence is preserved"
        )
    prompts = [
        _json(line) for line in _read(args.workload).splitlines() if line.strip()
    ]
    recipe = prepare_llamacpp_build_recipe(
        args.reference_bin_dir,
        args.candidate_bin_dir,
        args.model,
        args.model_sha256,
        prompts,
        threads=args.threads,
        context=args.context,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    if args.policy is not None:
        recipe = replace(recipe, policy=policy_from_data(_json(_read(args.policy))))
    output.parent.mkdir(parents=True, exist_ok=True)
    qualify_cpu_providers(recipe, qualifications)
    _write_atomic(output, study_recipe_to_json(recipe) + "\n")
    stdout.write(f"Prepared {output}\nRecipe: {study_recipe_digest(recipe)}\n")
    stdout.write("Both provider qualification records are saved beside the recipe.\n")
    return 0
