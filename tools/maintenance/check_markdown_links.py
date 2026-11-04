#!/usr/bin/env python3
"""Validate repository-local links in Markdown files."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
SKIP_PREFIXES = ("#", "http://", "https://", "mailto:", "data:")
SKIP_PARTS = {".git", ".pytest_cache", ".venv", "build", "dist"}


def _target_path(source: Path, raw_target: str) -> Path | None:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    if not target or target.startswith(SKIP_PREFIXES):
        return None

    # Drop an optional Markdown title and a fragment/query suffix.
    target = target.split(' "', 1)[0].split(" '", 1)[0]
    target = target.split("#", 1)[0].split("?", 1)[0]
    if not target:
        return None
    target = unquote(target)
    if target.startswith("/"):
        return None
    return (source.parent / target).resolve()


def _sources(root: Path) -> list[Path]:
    if not (root / ".git").exists():
        return sorted(root.rglob("*.md"))
    result: subprocess.CompletedProcess[bytes] | None
    try:
        result = subprocess.run(
            [
                "git",
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
                "--",
                "*.md",
            ],
            cwd=root,
            capture_output=True,
            check=False,
        )
    except OSError:
        result = None
    if result is not None and result.returncode == 0:
        return sorted(
            {
                root / name
                for name in result.stdout.decode("utf-8").split("\0")
                if name and (root / name).is_file()
            }
        )
    return sorted(root.rglob("*.md"))


def check(root: Path) -> list[tuple[Path, str]]:
    broken: list[tuple[Path, str]] = []
    for source in _sources(root):
        if any(part in SKIP_PARTS for part in source.parts):
            continue
        text = source.read_text(encoding="utf-8")
        for raw_target in LINK_RE.findall(text):
            target = _target_path(source, raw_target)
            if target is not None and not target.exists():
                broken.append((source.relative_to(root), raw_target))
    return broken


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    broken = check(root)
    if broken:
        for source, target in broken:
            print(f"{source}: {target}")
        print(f"\n{len(broken)} broken local Markdown link(s)", file=sys.stderr)
        return 1
    print("All local Markdown links are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
