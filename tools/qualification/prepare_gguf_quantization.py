"""Source-checkout wrapper for the installed GGUF quantization preparation CLI."""

from __future__ import annotations

import sys

from metria.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["recipe", "prepare-gguf-quantization", *sys.argv[1:]]))
