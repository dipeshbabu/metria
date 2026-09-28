"""Source compatibility alias for metria.fidelity.backends.sglang."""

import sys
from importlib import import_module

_implementation = import_module("metria.fidelity.backends.sglang")
sys.modules[__name__] = _implementation
