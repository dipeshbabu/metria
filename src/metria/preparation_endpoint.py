"""Prepare a ``remote_endpoint_change.v1`` recipe for two running servers.

The recipe names environment variables, never URLs or keys. Tokenizer files
are hashed from a local directory so prompts are tokenized identically for both
servers. ``comparison.vary`` is derived from the declared change, so any other
server-reported difference makes the runs not comparable.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

from .measurements import TokenTrajectoryProtocol
from .models import ComparisonPlan, RunSpec, StudySpec
from .recipes import StudyRecipe, study_recipe_digest, study_recipe_to_json
from .runtimes.endpoint import tokenizer_pins
from .runtimes.endpoint_identity import validate_declared

_TOKENIZER_CANDIDATES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
)


def hash_tokenizer(directory: str | Path) -> dict[str, str]:
    """Pin every known tokenizer file present in a local directory."""

    root = Path(directory).expanduser().resolve()
    pins = {}
    for name in _TOKENIZER_CANDIDATES:
        path = root / name
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(chunk)
            pins[name] = digest.hexdigest()
    return tokenizer_pins(pins)


def _scenario(
    max_tokens: int,
    seed: int,
    temperature: float,
    seeds: Sequence[int] | None,
    top_p: float | None,
    stop: Sequence[str] | None,
    chat_template: bool,
) -> dict[str, Any]:
    scenario: dict[str, Any] = {"max_tokens": max_tokens, "temperature": temperature}
    if temperature > 0:
        scenario["seeds"] = list(seeds or ())
    else:
        scenario["seed"] = seed
    if top_p is not None:
        scenario["top_p"] = top_p
    if stop:
        scenario["stop"] = list(stop)
    if chat_template:
        scenario["chat_template"] = True
    return scenario


def _collector_block(collector: Mapping[str, Any], role: str) -> dict[str, Any]:
    """One run's collector settings from the shared preparation options."""

    return {
        "url_env": collector[f"{role}_url_env"],
        "keys_env": collector["keys_env"],
        "deployment": collector[f"{role}_deployment"],
        "role": role,
        "trust": dict(collector.get("trust") or {}),
    }


def prepare_endpoint_recipe(
    *,
    model_id: str,
    tokenizer_dir: str | Path,
    prompts: Sequence[Mapping[str, Any]],
    reference_url_env: str,
    candidate_url_env: str,
    reference_declared: Mapping[str, Any],
    candidate_declared: Mapping[str, Any],
    policy: Any,
    reference_key_env: str | None = None,
    candidate_key_env: str | None = None,
    flavor: str = "vllm",
    max_tokens: int = 16,
    seed: int = 42,
    warmup_trials: int = 0,
    measured_trials: int = 2,
    timeout_s: float = 120,
    staged_distributions: bool = False,
    concurrency: int | None = None,
    concurrency_rounds: int = 2,
    temperature: float = 0.0,
    seeds: Sequence[int] | None = None,
    top_p: float | None = None,
    stop: Sequence[str] | None = None,
    chat_template: bool = False,
    collector: Mapping[str, Any] | None = None,
) -> StudyRecipe:
    from .verification_endpoint import (
        analyses,
        derived_vary,
        trial_identity,
        validate_endpoint_recipe,
    )

    config: dict[str, Any] = {
        "prompts": tuple(prompts),
        "warmup_trials": warmup_trials,
        "measured_trials": measured_trials,
    }
    if staged_distributions:
        config["staged_distributions"] = True
    if concurrency is not None:
        config["concurrency"] = {
            "max_in_flight": concurrency,
            "rounds": concurrency_rounds,
        }
    model = {
        "id": model_id,
        "tokenizer_path": str(Path(tokenizer_dir).expanduser().resolve()),
        "tokenizer_files": hash_tokenizer(tokenizer_dir),
    }

    scenario = _scenario(
        max_tokens, seed, temperature, seeds, top_p, stop, chat_template
    )

    def run(
        url_env: str, key_env: str | None, declared: Mapping[str, Any], role: str
    ) -> RunSpec:
        runtime: dict[str, Any] = {
            "name": "openai_endpoint",
            "flavor": flavor,
            "base_url_env": url_env,
            "timeout_s": timeout_s,
            "max_retries": 2,
            "declared": validate_declared(declared),
        }
        if key_env is not None:
            runtime["api_key_env"] = key_env
        if collector is not None:
            runtime["collector"] = _collector_block(collector, role)
        return RunSpec(
            model=model,
            runtime=runtime,
            scenario=scenario,
            measurements=(TokenTrajectoryProtocol.name,),
            trial_policy=trial_identity(config),
        )

    left = run(reference_url_env, reference_key_env, reference_declared, "reference")
    right = run(candidate_url_env, candidate_key_env, candidate_declared, "candidate")
    recipe = StudyRecipe(
        study=StudySpec(
            name="remote-endpoint-change",
            runs=(left, right),
            comparison=ComparisonPlan(
                vary=derived_vary(left, right),
                control=frozenset(
                    {"model", "scenario", "measurements", "trial_policy"}
                ),
                analyses=analyses(config),
            ),
        ),
        measurement_configs={TokenTrajectoryProtocol.name: config},
        environment={},
        policy=policy,
    )
    validate_endpoint_recipe(recipe)
    return recipe


def _declarations(values: Sequence[str] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in values or ():
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise ValueError("declarations use KEY=VALUE")
        result[key] = int(value) if value.isdecimal() else value
    return validate_declared(result)


def add_endpoint_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "prepare-endpoint",
        help="STAGED: compare two running OpenAI-compatible servers",
    )
    for name in ("model-id", "reference-url-env", "candidate-url-env"):
        parser.add_argument("--" + name, required=True)
    for name in ("tokenizer", "workload", "policy", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reference-key-env")
    parser.add_argument("--candidate-key-env")
    parser.add_argument(
        "--reference-declared",
        action="append",
        metavar="KEY=VALUE",
        help="operator statement about the reference deployment (repeatable)",
    )
    parser.add_argument("--candidate-declared", action="append", metavar="KEY=VALUE")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-trials", type=int, default=0)
    parser.add_argument("--measured-trials", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--staged-kld", action="store_true")
    parser.add_argument(
        "--concurrency",
        type=int,
        help="after sequential trials, send every prompt with this many requests "
        "in flight and measure batch invariance",
    )
    parser.add_argument("--concurrency-rounds", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--seeds",
        type=lambda text: [int(item) for item in text.split(",") if item],
        help="comma-separated seeds; required when --temperature is above 0",
    )
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--stop", action="append", help="stop sequence (repeatable)")
    parser.add_argument(
        "--chat-template",
        action="store_true",
        help="render the pinned tokenizer_config.json chat template locally",
    )
    group = parser.add_argument_group(
        "host evidence collectors",
        "require signed collector evidence from each server host (all or none)",
    )
    for name in ("reference-collector-url-env", "candidate-collector-url-env"):
        group.add_argument("--" + name)
    group.add_argument(
        "--collector-keys-env", help="variable holding the JSON key ring"
    )
    group.add_argument("--reference-deployment")
    group.add_argument("--candidate-deployment")
    group.add_argument(
        "--required-fact",
        action="append",
        default=[],
        help="fact that must be observed",
    )
    group.add_argument("--collector-max-age", type=float, default=300.0)


def _collector_options(args: Any) -> dict[str, Any] | None:
    names = (
        "reference_collector_url_env",
        "candidate_collector_url_env",
        "collector_keys_env",
        "reference_deployment",
        "candidate_deployment",
    )
    given = [getattr(args, name) for name in names]
    if not any(given):
        return None
    if not all(given):
        raise ValueError(
            "collector options need both collector URL variables, the key ring "
            "variable and both deployment names"
        )
    trust: dict[str, Any] = {"max_age_s": args.collector_max_age}
    if args.required_fact:
        trust.update(require_collector=True, required_facts=list(args.required_fact))
    return {
        "reference_url_env": args.reference_collector_url_env,
        "candidate_url_env": args.candidate_collector_url_env,
        "keys_env": args.collector_keys_env,
        "reference_deployment": args.reference_deployment,
        "candidate_deployment": args.candidate_deployment,
        "trust": trust,
    }


def prepare_endpoint_command(args: Any, stdout: TextIO) -> int:
    from .onboarding import _json, _read
    from .policies import policy_from_data
    from .verification import _write_atomic

    if args.output.exists():
        raise FileExistsError("recipe output already exists; choose a new file")
    prompts = [
        _json(line) for line in _read(args.workload).splitlines() if line.strip()
    ]
    recipe = prepare_endpoint_recipe(
        model_id=args.model_id,
        tokenizer_dir=args.tokenizer,
        prompts=prompts,
        reference_url_env=args.reference_url_env,
        candidate_url_env=args.candidate_url_env,
        reference_key_env=args.reference_key_env,
        candidate_key_env=args.candidate_key_env,
        reference_declared=_declarations(args.reference_declared),
        candidate_declared=_declarations(args.candidate_declared),
        policy=policy_from_data(_json(_read(args.policy))),
        max_tokens=args.max_tokens,
        seed=args.seed,
        warmup_trials=args.warmup_trials,
        measured_trials=args.measured_trials,
        timeout_s=args.timeout,
        staged_distributions=args.staged_kld,
        concurrency=args.concurrency,
        concurrency_rounds=args.concurrency_rounds,
        temperature=args.temperature,
        seeds=args.seeds,
        top_p=args.top_p,
        stop=args.stop,
        chat_template=args.chat_template,
        collector=_collector_options(args),
    )
    _write_atomic(args.output, study_recipe_to_json(recipe) + "\n")
    stdout.write(f"Prepared {args.output}\nRecipe: {study_recipe_digest(recipe)}\n")
    stdout.write(
        f"Set {args.reference_url_env} and {args.candidate_url_env} (and any key "
        "variables), then run: metria verify "
        f"{args.output} --output verification\n"
    )
    return 0
