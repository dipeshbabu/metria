#!/usr/bin/env python3
"""Verify required root files in the built Metria source distribution."""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path, PurePosixPath

_REQUIRED_ROOT_FILES = frozenset({"pyproject.toml", "LICENSE", "NOTICE"})


def check_sdist(path: Path) -> list[str]:
    """Return validation errors for one Metria ``.tar.gz`` source distribution."""

    errors: list[str] = []
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            members = tuple(member.name for member in archive.getmembers())
    except (OSError, tarfile.TarError) as exc:
        return [f"{path}: cannot read source distribution: {exc}"]

    roots = {
        PurePosixPath(name).parts[0]
        for name in members
        if PurePosixPath(name).parts
    }
    if len(roots) != 1:
        errors.append(
            f"{path}: expected exactly one top-level directory; found {sorted(roots)!r}"
        )
        return errors

    root = next(iter(roots))
    member_set = set(members)
    for filename in sorted(_REQUIRED_ROOT_FILES):
        expected = f"{root}/{filename}"
        if expected not in member_set:
            errors.append(
                f"{path}: missing required source-distribution file {expected}"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdists", nargs="+", type=Path)
    args = parser.parse_args()

    errors: list[str] = []
    for sdist in args.sdists:
        errors.extend(check_sdist(sdist))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Validated {len(args.sdists)} source distribution(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
