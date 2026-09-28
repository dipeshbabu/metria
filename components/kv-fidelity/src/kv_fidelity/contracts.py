"""Source compatibility alias for metria.fidelity.contracts."""

import sys
from importlib import import_module

_implementation = import_module("metria.fidelity.contracts")
sys.modules[__name__] = _implementation
