"""Command-line interface for versioned Metria recipes and inspection."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, TextIO

from .capabilities import inspect_model_geometry
from .hardware import capture_hardware_fingerprint
from .inspection import capability_inspection_to_mapping, inspect_run_capabilities
from .recipes import (
    STUDY_RECIPE_SCHEMA,
    StudyRecipe,
    load_study_recipe,
    study_recipe_digest,
    study_recipe_to_json,
)

INSPECTION_SCHEMA = "metria.inspection.v1"


def _parser() -> argparse.ArgumentParser:
    """Build the provisional Metria command-line parser."""

    parser = argparse.ArgumentParser(
        prog="metria",
        description="Evidence-oriented tooling for LLM inference studies.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="show the Metria core version and exit",
    )
    subparsers = parser.add_subparsers(dest="command")

    recipe = subparsers.add_parser(
        "recipe",
        help="validate and normalize versioned study recipes",
    )
    recipe_subparsers = recipe.add_subparsers(dest="recipe_command", required=True)

    validate = recipe_subparsers.add_parser(
        "validate",
        help="validate a recipe without executing it",
    )
    validate.add_argument("path", type=Path)
    validate.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="emit machine-readable validation metadata",
    )

    digest = recipe_subparsers.add_parser(
        "digest",
        help="print the canonical SHA-256 recipe digest",
    )
    digest.add_argument("path", type=Path)

    normalize = recipe_subparsers.add_parser(
        "normalize",
        help="write canonical human-readable JSON for a validated recipe",
    )
    normalize.add_argument("path", type=Path)
    normalize.add_argument(
        "-o",
        "--output",
        type=Path,
        help="write normalized JSON to a file instead of stdout",
    )

    inspect = subparsers.add_parser(
        "inspect",
        help="inspect requested model geometry, capabilities, and local hardware",
    )
    inspect.add_argument("path", type=Path, help="validated study recipe to inspect")
    inspect.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="emit machine-readable inspection evidence",
    )
    return parser


def _version() -> str:
    """Resolve the package version without creating an import cycle."""

    from . import __version__

    return __version__


def _load(path: Path) -> StudyRecipe:
    """Load one recipe and convert filesystem errors into concise CLI errors."""

    try:
        return load_study_recipe(path)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def _summary(recipe: StudyRecipe, path: Path) -> dict[str, object]:
    """Return non-sensitive structural validation metadata."""

    runtimes = sorted(
        {str(run.runtime.get("name", "<missing>")) for run in recipe.study.runs}
    )
    measurements = sorted(
        {measurement for run in recipe.study.runs for measurement in run.measurements}
    )
    return {
        "valid": True,
        "schema": STUDY_RECIPE_SCHEMA,
        "path": str(path),
        "study": recipe.study.name,
        "runs": len(recipe.study.runs),
        "runtimes": runtimes,
        "measurements": measurements,
        "analyses": list(recipe.study.comparison.analyses),
        "digest": study_recipe_digest(recipe),
    }


def _write_normalized(recipe: StudyRecipe, output: Path | None, stdout: TextIO) -> None:
    """Write normalized recipe JSON to stdout or an explicitly requested path."""

    text = study_recipe_to_json(recipe) + "\n"
    if output is None:
        stdout.write(text)
        return
    output.write_text(text, encoding="utf-8")


def _jsonable(value: Any) -> Any:
    """Convert immutable Metria evidence containers into strict JSON values."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, frozenset, set)):
        return [_jsonable(item) for item in value]
    return value


def _inspection_payload(recipe: StudyRecipe, path: Path) -> dict[str, Any]:
    """Build privacy-conscious data-only capability and hardware inspection."""

    run_payloads: list[dict[str, Any]] = []
    for index, run in enumerate(recipe.study.runs):
        geometry = inspect_model_geometry(run.model)
        capabilities = inspect_run_capabilities(run)
        model_identity = {
            key: run.model[key]
            for key in (
                "id",
                "revision",
                "tokenizer_id",
                "tokenizer_revision",
            )
            if key in run.model
        }
        run_payloads.append(
            {
                "index": index,
                "runtime": run.runtime.get("name"),
                "model": model_identity,
                "geometry": (
                    geometry.geometry.to_mapping()
                    if geometry.geometry is not None
                    else None
                ),
                **capability_inspection_to_mapping(capabilities),
            }
        )

    return {
        "schema": INSPECTION_SCHEMA,
        "path": str(path),
        "study": recipe.study.name,
        "recipe_digest": study_recipe_digest(recipe),
        "hardware": capture_hardware_fingerprint().to_mapping(),
        "runs": run_payloads,
    }


def _write_inspection_human(payload: Mapping[str, Any], stdout: TextIO) -> None:
    """Render a concise inspection summary without dumping sensitive recipe input."""

    hardware = payload["hardware"]
    assert isinstance(hardware, Mapping)
    platform = hardware["platform"]
    assert isinstance(platform, Mapping)
    stdout.write(
        f"study {payload['study']} {payload['recipe_digest']}\n"
        f"hardware {platform.get('system')} {platform.get('machine')}\n"
    )
    runs = payload["runs"]
    assert isinstance(runs, Sequence)
    for item in runs:
        assert isinstance(item, Mapping)
        stdout.write(
            f"run[{item['index']}] runtime={item.get('runtime')} "
            f"allowed={str(bool(item['allowed'])).lower()}\n"
        )
        capabilities = item["capabilities"]
        assert isinstance(capabilities, Mapping)
        for name in sorted(capabilities):
            capability = capabilities[name]
            assert isinstance(capability, Mapping)
            stdout.write(f"  {name}: {capability.get('status')}\n")


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the provisional CLI and return a process-style exit status."""

    out = stdout or sys.stdout
    err = stderr or sys.stderr
    parser = _parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.version:
        out.write(f"metria {_version()}\n")
        return 0
    if args.command is None:
        parser.print_help(file=err)
        return 2

    path: Path = args.path
    try:
        recipe = _load(path)
        if args.command == "inspect":
            payload = _inspection_payload(recipe, path)
            if args.json_output:
                out.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")
            else:
                _write_inspection_human(payload, out)
            return 0
        if args.command == "recipe" and args.recipe_command == "validate":
            summary = _summary(recipe, path)
            if args.json_output:
                out.write(json.dumps(summary, sort_keys=True) + "\n")
            else:
                out.write(
                    f"valid {STUDY_RECIPE_SCHEMA} "
                    f"{recipe.study.name} {summary['digest']}\n"
                )
            return 0
        if args.command == "recipe" and args.recipe_command == "digest":
            out.write(study_recipe_digest(recipe) + "\n")
            return 0
        if args.command == "recipe" and args.recipe_command == "normalize":
            _write_normalized(recipe, args.output, out)
            return 0
    except (OSError, TypeError, ValueError) as exc:
        err.write(f"metria: error: {exc}\n")
        return 2

    err.write("metria: error: unsupported command\n")
    return 2
