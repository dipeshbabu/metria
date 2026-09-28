"""Compatibility launcher for the library-backed TurboQuant diagnostic."""

from __future__ import annotations

import sys
from pathlib import Path

source_dir = Path(__file__).resolve().parents[2] / "src"
if (source_dir / "metria" / "processes.py").is_file():
    sys.path.insert(0, str(source_dir))
try:
    from metria.integrations import turboquant_diagnostic_runner as _implementation
except ModuleNotFoundError as exc:
    if exc.name is None or not exc.name.startswith("metria"):
        raise
    raise SystemExit("Run this diagnostic from a complete Metria checkout.") from None

if __name__ == "__main__":
    raise SystemExit(_implementation.main())
else:
    # Preserve the historical import path and monkeypatch surface for callers.
    sys.modules[__name__] = _implementation
