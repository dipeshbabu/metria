"""Enforce independent branch-aware coverage floors for high-risk modules."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def evaluate_gates(
    report: Mapping[str, Any], gates: Mapping[str, Any], scope: str
) -> tuple[list[dict[str, Any]], list[str]]:
    files = report.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("coverage JSON must contain file summaries")
    rows, errors = [], []
    for path, minimum in gates[scope].items():
        matches = [
            data
            for name, data in files.items()
            if name.replace("\\", "/") == path
            or name.replace("\\", "/").endswith("/" + path)
        ]
        if len(matches) != 1:
            errors.append(f"{path}: missing or ambiguous coverage evidence")
            continue
        summary = matches[0]["summary"]
        if report.get("meta", {}).get("branch_coverage") is not True:
            errors.append(f"{path}: branch coverage was not measured")
            continue
        counts = [
            summary.get(key)
            for key in (
                "covered_lines",
                "covered_branches",
                "num_statements",
                "num_branches",
            )
        ]
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts
        ):
            errors.append(f"{path}: invalid coverage counts")
            continue
        covered = summary["covered_lines"] + summary["covered_branches"]
        total = summary["num_statements"] + summary["num_branches"]
        if not total or covered > total:
            errors.append(f"{path}: no executable coverage evidence")
            continue
        percent = 100 * covered / total
        passed = percent >= minimum
        rows.append(
            {
                "module": path,
                "coverage": round(percent, 2),
                "minimum": minimum,
                "passed": passed,
            }
        )
        if not passed:
            errors.append(f"{path}: {percent:.2f}% is below its {minimum}% floor")
    if not rows and not errors:
        errors.append("no coverage gates were selected")
    return rows, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--scope", choices=("core", "components"), required=True)
    parser.add_argument(
        "--gates", type=Path, default=Path(__file__).with_name("coverage-gates.json")
    )
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    rows, errors = evaluate_gates(
        json.loads(args.report.read_text()),
        json.loads(args.gates.read_text()),
        args.scope,
    )
    lines = [
        "| Module | Branch-aware coverage | Floor | Result |",
        "|---|---:|---:|---|",
    ]
    lines.extend(
        f"| `{row['module']}` | {row['coverage']:.2f}% | {row['minimum']}% | {'pass' if row['passed'] else 'FAIL'} |"
        for row in rows
    )
    if errors:
        lines.extend(["", *errors])
    text = "\n".join(lines) + "\n"
    print(text)
    if args.summary:
        args.summary.write_text(text, encoding="utf-8")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
