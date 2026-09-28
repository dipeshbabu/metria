"""Source compatibility alias for metria.fidelity.backends.base."""

import sys
from importlib import import_module

_implementation = import_module("metria.fidelity.backends.base")
sys.modules[__name__] = _implementation
