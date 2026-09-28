"""Source compatibility alias for metria.fidelity.backends.vllm."""

import sys
from importlib import import_module

_implementation = import_module("metria.fidelity.backends.vllm")
sys.modules[__name__] = _implementation
