"""Source compatibility alias for metria.fidelity.measurement_math."""

import sys
from importlib import import_module

_implementation = import_module("metria.fidelity.measurement_math")
sys.modules[__name__] = _implementation
