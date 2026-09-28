"""Source compatibility alias for metria.fidelity.axes.plad."""

import sys
from importlib import import_module

_implementation = import_module("metria.fidelity.axes.plad")
sys.modules[__name__] = _implementation
