"""Source compatibility alias for metria.fidelity.score."""

import sys
from importlib import import_module

_implementation = import_module("metria.fidelity.score")
sys.modules[__name__] = _implementation
