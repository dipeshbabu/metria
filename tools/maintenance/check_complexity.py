"""Ratchet legacy size/complexity while requiring bounded new library functions."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

MAX_COMPLEXITY = 15
MAX_MODULE_LINES = 800
MAX_FUNCTION_LINES = 100


def _files(root: Path) -> list[Path]:
    folders = [
        root / "src",
        root / "tools",
        *sorted((root / "components").glob("*/src")),
    ]
    return sorted(path for folder in folders for path in folder.rglob("*.py"))


def _functions(tree: ast.AST, prefix: str = "") -> dict[int, tuple[str, int]]:
    found = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = prefix + node.name
            if not isinstance(node, ast.ClassDef):
                found[node.lineno] = (
                    name,
                    (node.end_lineno or node.lineno) - node.lineno + 1,
                )
            found.update(_functions(node, name + "."))
    return found


def measure(root: Path) -> dict[str, dict[str, Any]]:
    files = _files(root)
    result: dict[str, dict[str, Any]] = {}
    positions = {}
    for path in files:
        name = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        functions = _functions(ast.parse(text))
        positions[name] = functions
        result[name] = {
            "module_lines": sum(bool(line.strip()) for line in text.splitlines()),
            "function_lines": {name: length for name, length in functions.values()},
            "complexity": {},
        }
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--select",
            "C901",
            "--config",
            f"lint.mccabe.max-complexity={MAX_COMPLEXITY}",
            "--output-format",
            "json",
            *map(str, files),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode not in (0, 1):
        raise RuntimeError("Ruff complexity measurement failed: " + completed.stderr)
    for item in json.loads(completed.stdout):
        match = re.search(r"\((\d+) > \d+\)", item["message"])
        if item["code"] != "C901" or match is None:
            raise ValueError("unexpected Ruff complexity diagnostic")
        relative = Path(item["filename"]).resolve().relative_to(root).as_posix()
        name, _ = positions[relative][item["location"]["row"]]
        result[relative]["complexity"][name] = int(match.group(1))
    return result


def baseline_for(current: dict[str, dict[str, Any]]) -> dict[str, Any]:
    exceptions = {}
    for path, metrics in current.items():
        entry: dict[str, Any] = {}
        if metrics["module_lines"] > MAX_MODULE_LINES:
            entry["module_lines"] = metrics["module_lines"]
        for kind, target in (
            ("function_lines", MAX_FUNCTION_LINES),
            ("complexity", MAX_COMPLEXITY),
        ):
            selected = {
                name: value for name, value in metrics[kind].items() if value > target
            }
            if selected:
                entry[kind] = selected
        if entry:
            exceptions[path] = entry
    return {
        "schema": "metria.complexity_baseline.v1",
        "limits": {
            "complexity": MAX_COMPLEXITY,
            "module_lines": MAX_MODULE_LINES,
            "function_lines": MAX_FUNCTION_LINES,
        },
        "exceptions": exceptions,
    }


def regressions(
    current: dict[str, dict[str, Any]], baseline: dict[str, Any]
) -> list[str]:
    errors = []
    for path, metrics in current.items():
        old = baseline["exceptions"].get(path, {})
        if metrics["module_lines"] > max(MAX_MODULE_LINES, old.get("module_lines", 0)):
            errors.append(f"{path}: module grew beyond its size ceiling")
        for kind, target in (
            ("function_lines", MAX_FUNCTION_LINES),
            ("complexity", MAX_COMPLEXITY),
        ):
            for name, value in metrics[kind].items():
                if value > max(target, old.get(kind, {}).get(name, 0)):
                    errors.append(f"{path}:{name}: {kind} {value} exceeds its ceiling")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="maintainer-reviewed initialization only; never use to hide a regression",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    path = Path(__file__).with_name("complexity-baseline.json")
    current = measure(root)
    if args.write_baseline:
        path.write_text(
            json.dumps(baseline_for(current), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return 0
    errors = regressions(current, json.loads(path.read_text(encoding="utf-8")))
    print(
        "\n".join(errors)
        if errors
        else "Library/tool size and Ruff complexity ratchets passed."
    )
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
