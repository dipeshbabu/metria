"""Source compatibility alias for metria.fidelity.comparison."""

import sys
from importlib import import_module

_implementation = import_module("metria.fidelity.comparison")
sys.modules[__name__] = _implementation
