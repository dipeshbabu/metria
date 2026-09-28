from __future__ import annotations

import runpy
from pathlib import Path


def test_new_complexity_and_legacy_growth_are_rejected_but_reductions_pass():
    module = runpy.run_path(
        str(Path(__file__).parents[1] / "tools/maintenance/check_complexity.py")
    )
    original = {
        "legacy.py": {
            "module_lines": 900,
            "function_lines": {"run": 150},
            "complexity": {"run": 20},
        }
    }
    baseline = module["baseline_for"](original)
    assert not module["regressions"](original, baseline)
    smaller = {
        "legacy.py": {
            "module_lines": 800,
            "function_lines": {"run": 100},
            "complexity": {"run": 15},
        }
    }
    assert not module["regressions"](smaller, baseline)
    bigger = {
        "legacy.py": {
            "module_lines": 901,
            "function_lines": {"run": 151},
            "complexity": {"run": 21},
        }
    }
    assert len(module["regressions"](bigger, baseline)) == 3
    fresh = {
        "new.py": {
            "module_lines": 10,
            "function_lines": {"new": 10},
            "complexity": {"new": 16},
        }
    }
    assert module["regressions"](fresh, baseline)
