#!/usr/bin/env python3
"""Verify package boundaries and required data in built component wheels."""

from __future__ import annotations

import argparse
import sys
import zipfile
from email.parser import Parser
from pathlib import Path

KV_FIDELITY_PROJECT_URLS = {
    "Homepage, https://github.com/dipeshbabu/metria/tree/main/components/kv-fidelity",
    "Repository, https://github.com/dipeshbabu/metria",
    "Issues, https://github.com/dipeshbabu/metria/issues",
    "Changelog, https://github.com/dipeshbabu/metria/blob/main/components/kv-fidelity/CHANGELOG.md",
}


def _has_suffix(names: set[str], suffix: str) -> bool:
    return any(name.lower().endswith(suffix.lower()) for name in names)


def _read_single_member(path: Path, names: set[str], suffix: str) -> str | None:
    matches = [name for name in names if name.lower().endswith(suffix.lower())]
    if len(matches) != 1:
        return None
    with zipfile.ZipFile(path) as archive:
        return archive.read(matches[0]).decode("utf-8")


def check_wheel(path: Path) -> list[str]:
    errors: list[str] = []
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())

    if "kv_fidelity/__init__.py" in names:
        metadata_text = _read_single_member(path, names, ".dist-info/METADATA")
        if metadata_text is None:
            errors.append(f"{path}: expected exactly one METADATA file")
        else:
            metadata = Parser().parsestr(metadata_text)
            expected_fields = {
                "Name": "kv-fidelity",
                "License-Expression": "Apache-2.0",
                "Author": "Dipesh Tharu Mahato (dipeshbabu)",
            }
            for field, expected in expected_fields.items():
                actual = metadata.get(field)
                if actual != expected:
                    errors.append(
                        f"{path}: {field} is {actual!r}; expected {expected!r}"
                    )
            project_urls = set(metadata.get_all("Project-URL") or ())
            for project_url in sorted(KV_FIDELITY_PROJECT_URLS - project_urls):
                errors.append(f"{path}: missing Project-URL {project_url!r}")

        entry_points = _read_single_member(path, names, ".dist-info/entry_points.txt")
        if entry_points is None:
            errors.append(f"{path}: expected exactly one entry_points.txt file")
        else:
            if "kv-fidelity = kv_fidelity.cli:main" not in entry_points:
                errors.append(f"{path}: missing kv-fidelity console script")
            if any(
                line.partition("=")[0].strip() == "refract"
                for line in entry_points.splitlines()
            ):
                errors.append(f"{path}: unexpectedly contains refract console script")

        required = {
            "kv_fidelity/prompts/v0.1.jsonl",
            "kv_fidelity/prompts/README.md",
        }
        for name in sorted(required - names):
            errors.append(f"{path}: missing {name}")
        if not any(
            name.startswith("kv_fidelity/examples/") and name.endswith(".json")
            for name in names
        ):
            errors.append(f"{path}: missing packaged KV Fidelity JSON examples")
        if any(name.startswith("turboquant/") for name in names):
            errors.append(f"{path}: unexpectedly contains turboquant")
        if any(name.startswith("kv_fidelity/tests/") for name in names):
            errors.append(f"{path}: unexpectedly contains KV Fidelity tests")
    elif "turboquant/__init__.py" in names:
        if any(name.startswith("kv_fidelity/") for name in names):
            errors.append(f"{path}: unexpectedly contains kv_fidelity")
        if any("/tests/" in name or name.startswith("tests/") for name in names):
            errors.append(f"{path}: unexpectedly contains tests")
    else:
        errors.append(f"{path}: wheel contains neither component package")

    if any(name.startswith("refract/") for name in names):
        errors.append(f"{path}: unexpectedly contains refract")
    if not _has_suffix(names, "/LICENSE"):
        errors.append(f"{path}: missing LICENSE")
    if not _has_suffix(names, "/NOTICE"):
        errors.append(f"{path}: missing NOTICE")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", nargs="+", type=Path)
    args = parser.parse_args()

    errors: list[str] = []
    for wheel in args.wheels:
        errors.extend(check_wheel(wheel))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Validated {len(args.wheels)} component wheel(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
