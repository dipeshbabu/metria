"""Source compatibility aliases for Metria fidelity axes."""

import sys
from importlib import import_module

_implementation = import_module("metria.fidelity.axes")
for _name in ("gtm", "kld", "plad", "rniah", "trajectory"):
    sys.modules[f"{__name__}.{_name}"] = import_module(f"metria.fidelity.axes.{_name}")
sys.modules[__name__] = _implementation
