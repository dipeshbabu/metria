"""Reject contributor-local home paths in tracked documents and evidence."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_PATTERN = re.compile(
    r"(?:[A-Za-z]:[\\/]+Users[\\/]+|/Users/|/home/)([A-Za-z0-9_.-]+)|/root/"
)
_PLACEHOLDERS = {"user", "username", "your_user", "your_username", "example", "public"}
_SUFFIXES = {
    ".md",
    ".json",
    ".jsonl",
    ".txt",
    ".log",
    ".toml",
    ".yaml",
    ".yml",
    ".html",
}


def local_path_lines(text: str) -> list[int]:
    return [
        number
        for number, line in enumerate(text.splitlines(), start=1)
        if any(
            match.group(1) is None or match.group(1).lower() not in _PLACEHOLDERS
            for match in _PATTERN.finditer(line)
        )
    ]


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    tracked = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
        .decode("utf-8")
        .split("\0")
    )
    errors = []
    for name in tracked:
        path = root / name
        if path.suffix.lower() not in _SUFFIXES or not path.is_file():
            continue
        for line in local_path_lines(
            path.read_text(encoding="utf-8", errors="replace")
        ):
            errors.append(
                f"{name}:{line}: replace contributor-local home path with a portable placeholder or immutable reference"
            )
    if errors:
        print("\n".join(errors))
        return 1
    print("Tracked documents and evidence contain no contributor-local home paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
