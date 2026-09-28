"""Source compatibility alias for metria.fidelity.backends.llamacpp."""

import sys
from importlib import import_module

_implementation = import_module("metria.fidelity.backends.llamacpp")
sys.modules[__name__] = _implementation
