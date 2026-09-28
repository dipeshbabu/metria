"""Source compatibility aliases for Metria fidelity backends."""

import sys
from importlib import import_module

_implementation = import_module("metria.fidelity.backends")
for _name in ("base", "llamacpp", "mlx", "sglang", "vllm"):
    sys.modules[f"{__name__}.{_name}"] = import_module(
        f"metria.fidelity.backends.{_name}"
    )
sys.modules[__name__] = _implementation
