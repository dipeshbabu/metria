"""Source compatibility alias for metria.fidelity.report."""

import sys
from importlib import import_module

_implementation = import_module("metria.fidelity.report")
sys.modules[__name__] = _implementation
