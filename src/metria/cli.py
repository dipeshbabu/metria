"""Command-line interface for versioned Metria recipes, records, and inspection."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from contextlib import redirect_stderr, redirect_stdout
from enum import Enum
from itertools import combinations
from pathlib import Path
from typing import Any, TextIO

from .capabilities import inspect_model_geometry
from .comparison import compare_runs
from .hardware import capture_hardware_fingerprint
from .inspection import capability_inspection_to_mapping, inspect_run_capabilities
from .models import CompatibilityIssue, CompatibilityReport, RunRecord
from .recipes import (
    STUDY_RECIPE_SCHEMA,
    StudyRecipe,
    load_study_recipe,
    study_recipe_digest,
    study_recipe_to_json,
)
from .records import load_run_record, run_evidence_digest, run_record_digest
from .verification import render_verification, verify_recipe

INSPECTION_SCHEMA = "metria.inspection.v1"
COMPARISON_REPORT_SCHEMA = "metria.comparison_report.v1"


def _parser() -> argparse.ArgumentParser:
    """Build the provisional Metria command-line parser."""

    parser = argparse.ArgumentParser(
        prog="metria",
        description="Verify reference/candidate LLM inference changes with retained evidence.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="show the Metria core version and exit",
    )
    subparsers = parser.add_subparsers(dest="command")

    _add_recipe_parser(subparsers)
    demo = subparsers.add_parser(
        "demo", help="run an explicitly synthetic verifier example without a model"
    )
    demo.add_argument(
        "--case", choices=("pass", "fail", "not-comparable"), default="pass"
    )
    demo.add_argument(
        "--output", type=Path, required=True, help="new synthetic evidence directory"
    )
    subparsers.add_parser(
        "fidelity",
        help="expert fidelity scoring and saved-report tools",
        add_help=False,
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

    compare = subparsers.add_parser(
        "compare",
        help="compare saved run records under an explicit study comparison plan",
    )
    compare.add_argument(
        "records",
        type=Path,
        nargs="+",
        help="two or more metria.run_record.v1 files",
    )
    compare.add_argument(
        "--recipe",
        type=Path,
        required=True,
        help="study recipe whose ComparisonPlan governs comparability",
    )
    compare.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="emit a machine-readable pairwise comparison report",
    )
    verify = subparsers.add_parser(
        "verify", help="execute a qualified reference/candidate inference change"
    )
    verify.add_argument("path", type=Path, help="reference/candidate study recipe")
    verify.add_argument(
        "--output",
        type=Path,
        default=Path("verification"),
        help="new evidence directory",
    )
    verify.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="emit the verification manifest as JSON",
    )
    return parser


def _add_recipe_parser(subparsers: Any) -> None:
    recipe = subparsers.add_parser(
        "recipe",
        help="validate and normalize versioned study recipes",
    )
    recipe_subparsers = recipe.add_subparsers(dest="recipe_command", required=True)
    _add_preparation_parser(recipe_subparsers)

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


def _add_preparation_parser(subparsers: Any) -> None:
    prepare = subparsers.add_parser(
        "prepare-vllm", help="prepare a pinned local vLLM prefix-cache recipe"
    )
    prepare.add_argument(
        "--model",
        type=Path,
        required=True,
        help="existing local model/tokenizer directory",
    )
    prepare.add_argument(
        "--descriptor", type=Path, help="trusted model file-to-SHA256 manifest"
    )
    prepare.add_argument(
        "--workload", type=Path, help="JSONL prompt rows with optional task checks"
    )
    prepare.add_argument(
        "--example",
        action="store_true",
        help="use bundled SmolLM2 descriptor/workload defaults; no model download",
    )
    prepare.add_argument("--policy", type=Path, help="explicit acceptance policy JSON")
    prepare.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new recipe file; contains supplied workload text",
    )
    prepare.add_argument("--context", type=int, default=512)
    prepare.add_argument("--max-tokens", type=int, default=16)
    prepare.add_argument("--warmup-trials", type=int, default=1)
    prepare.add_argument("--measured-trials", type=int, default=3)
    prepare.add_argument(
        "--timeout",
        type=float,
        default=900,
        help="whole-verification deadline in seconds",
    )


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


def _load_record(path: Path) -> RunRecord:
    """Load one saved record with its source path in diagnostics."""

    try:
        return load_run_record(path)
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


def _validate_record_plan_binding(
    record: RunRecord,
    recipe: StudyRecipe,
    *,
    path: Path,
) -> None:
    """Require a saved record to belong to the recipe supplying its comparison plan."""

    if record.study_name != recipe.study.name:
        raise ValueError(
            f"{path}: record study {record.study_name!r} does not match "
            f"comparison recipe study {recipe.study.name!r}"
        )
    if record.requested not in recipe.study.runs:
        raise ValueError(
            f"{path}: requested RunSpec is not present in comparison recipe "
            f"{recipe.study.name!r}"
        )


def _compatibility_issue_to_data(issue: CompatibilityIssue) -> dict[str, Any]:
    """Serialize one compatibility issue or retained waived difference."""

    return {
        "dimension": issue.dimension,
        "left": _jsonable(issue.left),
        "right": _jsonable(issue.right),
        "reason": issue.reason,
    }


def _compatibility_to_data(report: CompatibilityReport) -> dict[str, Any]:
    return {
        "compatible": report.compatible,
        "issues": [_compatibility_issue_to_data(issue) for issue in report.issues],
        "comparable_metrics": list(report.comparable_metrics),
        "incompatible_metrics": dict(report.incompatible_metrics),
        "waived_differences": [
            _compatibility_issue_to_data(issue) for issue in report.waived_differences
        ],
    }


def _comparison_payload(
    recipe: StudyRecipe,
    recipe_path: Path,
    loaded: Sequence[tuple[Path, RunRecord]],
) -> dict[str, Any]:
    """Compare every saved record pair under one explicit study plan."""

    records: list[dict[str, Any]] = []
    for path, record in loaded:
        _validate_record_plan_binding(record, recipe, path=path)
        records.append(
            {
                "path": str(path),
                "run_id": record.run_id,
                "record_digest": run_record_digest(record),
                "evidence_digest": run_evidence_digest(record),
            }
        )

    pairs: list[dict[str, Any]] = []
    for (left_path, left), (right_path, right) in combinations(loaded, 2):
        pairs.append(
            {
                "left": {"path": str(left_path), "run_id": left.run_id},
                "right": {"path": str(right_path), "run_id": right.run_id},
                "report": _compatibility_to_data(
                    compare_runs(left, right, recipe.study.comparison)
                ),
            }
        )

    return {
        "schema": COMPARISON_REPORT_SCHEMA,
        "recipe": {
            "path": str(recipe_path),
            "study": recipe.study.name,
            "digest": study_recipe_digest(recipe),
        },
        "records": records,
        "pairs": pairs,
        "compatible": all(pair["report"]["compatible"] for pair in pairs),
    }


def _write_comparison_human(payload: Mapping[str, Any], stdout: TextIO) -> None:
    """Render pairwise comparability without hiding metric-level incompatibility."""

    pairs = payload["pairs"]
    assert isinstance(pairs, Sequence)
    for pair in pairs:
        assert isinstance(pair, Mapping)
        left = pair["left"]
        right = pair["right"]
        report = pair["report"]
        assert isinstance(left, Mapping)
        assert isinstance(right, Mapping)
        assert isinstance(report, Mapping)
        stdout.write(
            f"compare {left['run_id']} {right['run_id']} "
            f"compatible={str(bool(report['compatible'])).lower()}\n"
        )
        comparable = report["comparable_metrics"]
        assert isinstance(comparable, Sequence)
        if comparable:
            stdout.write("  comparable_metrics: " + ", ".join(comparable) + "\n")
        issues = report["issues"]
        assert isinstance(issues, Sequence)
        for issue in issues:
            assert isinstance(issue, Mapping)
            stdout.write(f"  issue {issue['dimension']}: {issue['reason']}\n")
        waived = report["waived_differences"]
        assert isinstance(waived, Sequence)
        for issue in waived:
            assert isinstance(issue, Mapping)
            stdout.write(f"  waived {issue['dimension']}: {issue['reason']}\n")
        incompatible = report["incompatible_metrics"]
        assert isinstance(incompatible, Mapping)
        for name in sorted(incompatible):
            stdout.write(f"  metric {name}: {incompatible[name]}\n")


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the provisional CLI and return a process-style exit status."""

    out = stdout or sys.stdout
    err = stderr or sys.stderr
    arguments = list(argv) if argv is not None else sys.argv[1:]
    if arguments[:1] == ["fidelity"]:
        return _fidelity_command(arguments[1:], out, err)
    return _main(arguments, out, err)


def _fidelity_command(argv: Sequence[str], out: TextIO, err: TextIO) -> int:
    from .fidelity.cli import main as fidelity_main

    with redirect_stdout(out), redirect_stderr(err):
        try:
            return fidelity_main(list(argv), prog="metria fidelity", version=_version())
        except SystemExit as exc:
            return int(exc.code or 0)


def _main(argv: Sequence[str], out: TextIO, err: TextIO) -> int:
    parser = _parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.version:
        out.write(f"metria {_version()}\n")
        return 0
    if args.command is None:
        parser.print_help(file=err)
        return 2

    try:
        if args.command == "verify":
            return _verify_command(args, out, err)
        if args.command == "demo":
            from .onboarding import demo_command

            return demo_command(args, out, err)
        if args.command == "recipe" and args.recipe_command == "prepare-vllm":
            from .onboarding import prepare_command

            return prepare_command(args, out)
        if args.command == "compare":
            if len(args.records) < 2:
                raise ValueError("compare requires at least two run record files")
            recipe = _load(args.recipe)
            loaded = tuple((path, _load_record(path)) for path in args.records)
            payload = _comparison_payload(recipe, args.recipe, loaded)
            if args.json_output:
                out.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")
            else:
                _write_comparison_human(payload, out)
            return 0 if payload["compatible"] else 1

        path: Path = args.path
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
        if args.command == "verify":
            return _verification_error(
                exc, "INVALID_CONFIGURATION", 2, args.json_output, out, err
            )
        err.write(f"metria: error: {exc}\n")
        return 2

    err.write("metria: error: unsupported command\n")
    return 2


def _verification_error(
    error: Exception,
    status: str,
    code: int,
    json_output: bool,
    out: TextIO,
    err: TextIO,
) -> int:
    """Return a machine-readable failure without exposing exception values."""
    payload = {
        "schema": "metria.verification_error.v1",
        "verdict": status,
        "exit_code": code,
        "error_type": type(error).__name__,
        "message_sha256": hashlib.sha256(str(error).encode("utf-8")).hexdigest(),
    }
    if json_output:
        out.write(json.dumps(payload, sort_keys=True) + "\n")
        err.write(
            f"metria: {status} ({type(error).__name__}); no complete verification bundle was published\n"
        )
    else:
        err.write(f"metria: error: {error}\n")
    return code


def _verify_command(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    recipe = _load(args.path)
    try:
        result = verify_recipe(recipe, args.output)
    except OSError as exc:
        return _verification_error(
            exc, "EXECUTION_FAILED", 5, args.json_output, out, err
        )
    if args.json_output:
        out.write(json.dumps(result.to_data(), sort_keys=True, allow_nan=False) + "\n")
    else:
        out.write(render_verification(result.manifest))
        out.write(f"Evidence: {result.output_dir}\n")
    return result.exit_code
