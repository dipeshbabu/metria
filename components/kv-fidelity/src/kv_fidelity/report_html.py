"""Source compatibility alias for metria.fidelity.report_html."""

import sys
from importlib import import_module

_implementation = import_module("metria.fidelity.report_html")
sys.modules[__name__] = _implementation
