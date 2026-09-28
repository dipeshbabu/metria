"""Source compatibility alias for metria.fidelity.axes.trajectory."""

import sys
from importlib import import_module

_implementation = import_module("metria.fidelity.axes.trajectory")
sys.modules[__name__] = _implementation
