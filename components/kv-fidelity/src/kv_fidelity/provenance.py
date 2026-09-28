"""Source compatibility alias for metria.fidelity.provenance."""

import sys
from importlib import import_module

_implementation = import_module("metria.fidelity.provenance")
sys.modules[__name__] = _implementation
