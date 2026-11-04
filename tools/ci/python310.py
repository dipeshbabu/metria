# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = [
#   "metria",
#   "kv-fidelity",
#   "turboquant-reference",
#   "pytest>=9.0.3",
#   "pytest-cov>=4.0",
#   "coverage>=7.16.2",
#   "mypy==2.4.1",
#   "ruff>=0.17.3",
#   "rich>=14.0",
#   "types-requests>=2.33.0.20261030",
# ]
# [tool.uv.sources]
# metria = { path = "../..", editable = true }
# kv-fidelity = { path = "../../components/kv-fidelity", editable = true }
# turboquant-reference = { path = "../../components/turboquant-reference", editable = true }
# ///
"""Keep core Python 3.10 checks isolated from native/backend maintenance stacks."""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in {"python", "pytest", "mypy", "ruff"}:
        raise SystemExit("expected python, pytest, mypy, or ruff and its arguments")
    tool, *arguments = sys.argv[1:]
    command = [sys.executable]
    if tool != "python":
        command.extend(("-m", tool))
    return subprocess.run([*command, *arguments], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
