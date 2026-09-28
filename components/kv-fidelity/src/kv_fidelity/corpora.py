"""Source compatibility alias for metria.fidelity.corpora."""

import sys
from importlib import import_module

_implementation = import_module("metria.fidelity.corpora")
sys.modules[__name__] = _implementation
