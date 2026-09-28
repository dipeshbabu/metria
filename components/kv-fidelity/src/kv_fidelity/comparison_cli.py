"""Source compatibility alias for metria.fidelity.comparison_cli."""

import sys
from importlib import import_module

_implementation = import_module("metria.fidelity.comparison_cli")
sys.modules[__name__] = _implementation
