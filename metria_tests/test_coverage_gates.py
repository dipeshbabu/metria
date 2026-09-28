from __future__ import annotations

import runpy
from pathlib import Path


def test_well_covered_module_cannot_mask_a_weak_runtime():
    check = runpy.run_path(
        str(Path(__file__).parents[1] / "tools/maintenance/check_coverage_gates.py")
    )["evaluate_gates"]
    report = {
        "meta": {"branch_coverage": True},
        "files": {
            "core.py": {
                "summary": {
                    "covered_lines": 1000,
                    "num_statements": 1000,
                    "covered_branches": 1000,
                    "num_branches": 1000,
                }
            },
            "runtime.py": {
                "summary": {
                    "covered_lines": 1,
                    "num_statements": 10,
                    "covered_branches": 1,
                    "num_branches": 10,
                }
            },
        },
    }
    rows, errors = check(report, {"core": {"core.py": 90, "runtime.py": 80}}, "core")
    assert rows[0]["passed"] and not rows[1]["passed"]
    assert errors and "runtime.py" in errors[0]


def test_missing_module_or_line_only_coverage_fails_closed():
    check = runpy.run_path(
        str(Path(__file__).parents[1] / "tools/maintenance/check_coverage_gates.py")
    )["evaluate_gates"]
    assert check({"files": {}}, {"core": {"missing.py": 80}}, "core")[1]
    report = {"meta": {"branch_coverage": False}, "files": {"one.py": {"summary": {}}}}
    assert "branch coverage" in check(report, {"core": {"one.py": 80}}, "core")[1][0]
