"""Installed-package preparation and explicit synthetic demonstrations."""

from __future__ import annotations

import importlib.metadata
import json
import os
import shlex
import subprocess
import sys
from importlib import resources
from pathlib import Path
from typing import Any, TextIO

from .measurements.prefix_workload import trajectory_config
from .policies import policy_from_data
from .preparation import prepare_vllm_prefix_recipe
from .processes import run_process
from .recipes import _unique_json_object, study_recipe_digest, study_recipe_to_json


def _reject_constant(value: str) -> None:
    raise ValueError("JSON input must contain only finite numbers")


def _json(text: str) -> Any:
    return json.loads(
        text, object_pairs_hook=_unique_json_object, parse_constant=_reject_constant
    )


def _read(path: Path) -> str:
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("preparation inputs must be at most 16 MiB")
    return path.read_text(encoding="utf-8")


def _runtime_host_supported() -> bool:
    return sys.platform == "linux"


def prepare_command(args: Any, stdout: TextIO) -> int:
    from dataclasses import replace

    from .verification import _write_atomic

    if args.output.exists():
        raise FileExistsError("recipe output already exists; choose a new file")
    if not args.example and (args.descriptor is None or args.workload is None):
        raise ValueError(
            "provide --descriptor and --workload, or use --example for the pinned SmolLM2 fixture"
        )
    assets = resources.files("metria").joinpath("data")
    descriptor = _json(
        _read(args.descriptor)
        if args.descriptor
        else assets.joinpath("smollm2-135m.json").read_text(encoding="utf-8")
    )
    if not isinstance(descriptor, dict):
        raise ValueError(
            "model descriptor must be a JSON object with trusted file pins"
        )
    text = (
        _read(args.workload)
        if args.workload
        else assets.joinpath("prefix-workload.jsonl").read_text(encoding="utf-8")
    )
    prompts = [_json(line) for line in text.splitlines() if line.strip()]
    trajectory_config(
        {
            "prompts": prompts,
            "warmup_trials": args.warmup_trials,
            "measured_trials": args.measured_trials,
        }
    )
    if not _runtime_host_supported():
        raise ValueError(
            "run preparation from a qualified vLLM environment on Linux (including WSL); this host is not qualified for native vLLM verification"
        )
    try:
        recipe = prepare_vllm_prefix_recipe(
            args.model,
            descriptor,
            prompts,
            context=args.context,
            max_tokens=args.max_tokens,
            warmup_trials=args.warmup_trials,
            measured_trials=args.measured_trials,
            timeout_s=args.timeout,
        )
    except importlib.metadata.PackageNotFoundError as exc:
        raise ValueError(
            "run preparation from a qualified vLLM environment with its tokenizer dependencies installed; base Metria does not install inference engines"
        ) from exc
    if args.policy is not None:
        recipe = replace(recipe, policy=policy_from_data(_json(_read(args.policy))))
    _write_atomic(args.output, study_recipe_to_json(recipe) + "\n")
    stdout.write(f"Prepared {args.output}\nRecipe: {study_recipe_digest(recipe)}\n")
    command = ["metria", "verify", str(args.output), "--output", "verification"]
    display = (
        subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)
    )
    stdout.write(f"Next: {display}\n")
    return 0


def demo_command(args: Any, stdout: TextIO, stderr: TextIO) -> int:
    if args.output.exists():
        raise FileExistsError("demo output already exists; choose a new directory")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    result = run_process(
        [
            sys.executable,
            "-m",
            "metria._demo_worker",
            "--case",
            args.case,
            "--output",
            str(args.output.resolve()),
        ],
        timeout_s=60,
        env=environment,
        max_output_bytes=1024 * 1024,
    )
    stdout.write(result.stdout)
    if result.stderr:
        stderr.write(result.stderr)
    if result.timed_out:
        stderr.write("synthetic demo exceeded its execution deadline\n")
        return 5
    try:
        evidence = _json(
            (args.output / "verification.json").read_text(encoding="utf-8")
        )
        if (
            evidence.get("fixture_only") is not True
            or evidence.get("scope") != "synthetic_fixture.v1"
            or evidence.get("exit_code") != result.returncode
        ):
            raise ValueError("demo result is not correctly marked as synthetic")
    except (OSError, ValueError, TypeError, AttributeError):
        stderr.write("synthetic demo did not produce a valid, labelled result bundle\n")
        return 5
    return result.returncode
