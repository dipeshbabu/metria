"""Source compatibility alias for metria.fidelity.axes.kld."""

import sys
from importlib import import_module

_implementation = import_module("metria.fidelity.axes.kld")
sys.modules[__name__] = _implementation
