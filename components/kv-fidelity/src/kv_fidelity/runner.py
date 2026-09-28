"""Source compatibility alias for metria.fidelity.runner."""

import sys
from importlib import import_module

_implementation = import_module("metria.fidelity.runner")
sys.modules[__name__] = _implementation
