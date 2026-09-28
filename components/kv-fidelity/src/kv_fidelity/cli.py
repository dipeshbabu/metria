"""Source compatibility alias for metria.fidelity.cli."""

import sys
from importlib import import_module

_implementation = import_module("metria.fidelity.cli")
if __name__ == "__main__":
    raise SystemExit(_implementation.main())
sys.modules[__name__] = _implementation
