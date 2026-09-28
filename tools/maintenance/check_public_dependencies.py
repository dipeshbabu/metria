"""Require stable, non-yanked public dependencies before component publication."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version

tomllib = importlib.import_module("tomllib" if sys.version_info >= (3, 11) else "tomli")


def compatible_releases(
    requirement: Requirement, data: Mapping[str, Any]
) -> tuple[str, ...]:
    releases = data.get("releases")
    if not isinstance(releases, Mapping):
        raise ValueError("package index response does not contain releases")
    available = []
    for raw_version, files in releases.items():
        try:
            version = Version(raw_version)
        except InvalidVersion:
            continue
        if (
            version.is_prerelease
            or version.is_devrelease
            or not requirement.specifier.contains(version, prereleases=False)
        ):
            continue
        if isinstance(files, list) and any(
            isinstance(item, Mapping) and item.get("yanked") is False for item in files
        ):
            available.append(version)
    return tuple(str(version) for version in sorted(available))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    project = tomllib.loads(args.manifest.read_text(encoding="utf-8"))["project"]
    dependencies = [Requirement(value) for value in project.get("dependencies", [])]
    metria = [
        requirement
        for requirement in dependencies
        if requirement.name.lower() == "metria"
    ]
    if len(metria) != 1:
        raise ValueError("KV Fidelity must declare exactly one Metria dependency")
    request = urllib.request.Request(
        "https://pypi.org/pypi/metria/json", headers={"Accept": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        data = json.load(response)
    available = compatible_releases(metria[0], data)
    if not available:
        raise SystemExit(
            f"Publication blocked: {metria[0]} has no compatible stable non-yanked release on PyPI. Publish and verify the required Metria release first."
        )
    print(
        f"Public dependency prerequisite satisfied: metria {available[-1]} matches {metria[0]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
