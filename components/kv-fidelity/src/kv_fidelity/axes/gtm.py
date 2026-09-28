"""Source compatibility alias for metria.fidelity.axes.gtm."""

import sys
from importlib import import_module

_implementation = import_module("metria.fidelity.axes.gtm")
sys.modules[__name__] = _implementation
