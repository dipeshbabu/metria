"""Source-checkout wrapper for installed CPU runtime-upgrade preparation."""

from __future__ import annotations

import sys

from metria.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["recipe", "prepare-vllm-upgrade", *sys.argv[1:]]))
