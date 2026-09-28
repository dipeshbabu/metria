"""Source compatibility alias for metria.fidelity.backends.mlx."""

import sys
from importlib import import_module

_implementation = import_module("metria.fidelity.backends.mlx")
sys.modules[__name__] = _implementation
