from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from metria import load_study_recipe
from metria.records import load_run_record


@pytest.mark.parametrize(
    "case,expected_code,verdict",
    [("pass", 0, "PASS"), ("fail", 1, "FAIL"), ("not-comparable", 3, "NOT_COMPARABLE")],
)
def test_copyable_fixture_runs_the_full_verifier_and_labels_synthetic_evidence(
    tmp_path, case, expected_code, verdict
):
    root = Path(__file__).parents[1]
    fixture = runpy.run_path(str(root / "examples/verification/run_fixture.py"))
    output = tmp_path / case
    assert fixture["run_fixture"](case, output) == expected_code
    data = json.loads((output / "verification.json").read_text())
    assert data["verdict"] == verdict
    assert data["fixture_only"] is True
    assert (
        "no real runtime or model qualification" in (output / "report.md").read_text()
    )
    for role in ("reference", "candidate"):
        record = load_run_record(output / f"{role}.run.json")
        assert record.provenance["verification"]["scope"] == "synthetic_fixture.v1"
    if case == "not-comparable":
        assert data["policy"]["status"] == "NOT_EVALUATED"
        assert not data["performance"]["available"]


def test_staged_catalog_recipes_validate_without_advertising_runtime_support():
    folder = Path(__file__).parents[1] / "examples/verification"
    catalog = json.loads((folder / "catalog.json").read_text())
    staged = [entry for entry in catalog["examples"] if entry["status"] == "staged"]
    assert len(staged) == 4
    for entry in staged:
        recipe = load_study_recipe(folder / entry["recipe"])
        assert len(recipe.study.runs) == 2
        assert recipe.study.comparison.vary
        assert entry["blocker"]
        assert recipe.study.name.startswith("STAGED-")
