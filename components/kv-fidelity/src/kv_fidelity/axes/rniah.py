"""Source compatibility alias for metria.fidelity.axes.rniah."""

import sys
from importlib import import_module

_implementation = import_module("metria.fidelity.axes.rniah")
sys.modules[__name__] = _implementation
