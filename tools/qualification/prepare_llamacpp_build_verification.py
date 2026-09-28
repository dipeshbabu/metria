"""Source-checkout wrapper for the installed build-comparison preparation CLI."""

from __future__ import annotations

import sys

from metria.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["recipe", "prepare-llamacpp-build", *sys.argv[1:]]))
