"""Minimal command-line interface for versioned Metria study recipes."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from .recipes import (
    STUDY_RECIPE_SCHEMA,
    load_study_recipe,
    study_recipe_digest,
    study_recipe_to_json,
)


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
    return parser


def _version() -> str:
    """Resolve the package version without creating an import cycle."""

    from . import __version__

    return __version__


def _load(path: Path):
    """Load one recipe and convert filesystem errors into concise CLI errors."""

    try:
        return load_study_recipe(path)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def _summary(recipe, path: Path) -> dict[str, object]:
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


def _write_normalized(recipe, output: Path | None, stdout: TextIO) -> None:
    """Write normalized recipe JSON to stdout or an explicitly requested path."""

    text = study_recipe_to_json(recipe) + "\n"
    if output is None:
        stdout.write(text)
        return
    output.write_text(text, encoding="utf-8")


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
        if args.recipe_command == "validate":
            summary = _summary(recipe, path)
            if args.json_output:
                out.write(json.dumps(summary, sort_keys=True) + "\n")
            else:
                out.write(
                    f"valid {STUDY_RECIPE_SCHEMA} "
                    f"{recipe.study.name} {summary['digest']}\n"
                )
            return 0
        if args.recipe_command == "digest":
            out.write(study_recipe_digest(recipe) + "\n")
            return 0
        if args.recipe_command == "normalize":
            _write_normalized(recipe, args.output, out)
            return 0
    except (OSError, TypeError, ValueError) as exc:
        err.write(f"metria: error: {exc}\n")
        return 2

    err.write("metria: error: unsupported command\n")
    return 2
