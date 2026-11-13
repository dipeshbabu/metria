"""User-supplied JSON fails as invalid input, never as an uncaught exception."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from metria.cli import main
from metria.onboarding import _json
from metria.recipes import load_study_recipe
from metria.records import load_run_record

REPO = Path(__file__).resolve().parents[1]
RECIPE = REPO / "artifacts/qualification/installed-workflow/study.json"
RECORD = REPO / "artifacts/qualification/installed-workflow/reference.run.json"
TOO_DEEP = "[" * 100_000 + "]" * 100_000


def _nest(document: str, depth: int) -> str:
    return document.replace('"__DEEP__"', "[" * depth + "]" * depth)


def test_deeply_nested_recipe_and_record_are_input_errors(tmp_path: Path) -> None:
    # Whether the JSON decoder itself recurses too deeply depends on the Python
    # build (3.14 on Linux parses this); either way it must be an input error.
    path = tmp_path / "deep.json"
    path.write_text(TOO_DEEP, encoding="utf-8")
    with pytest.raises((ValueError, TypeError)):
        load_study_recipe(path)
    with pytest.raises((ValueError, TypeError)):
        load_run_record(path)
    try:
        _json(TOO_DEEP)
    except ValueError as exc:
        assert "nested too deeply" in str(exc)


def test_recursion_during_parsing_is_reported_as_nesting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import metria.onboarding as onboarding
    import metria.recipes as recipes
    import metria.records as records

    def recurse(*args, **kwargs):
        raise RecursionError("maximum recursion depth exceeded")

    path = tmp_path / "any.json"
    path.write_text("{}", encoding="utf-8")
    for module in (recipes, records, onboarding):
        monkeypatch.setattr(module.json, "loads", recurse)
    with pytest.raises(ValueError, match="nested too deeply"):
        load_study_recipe(path)
    with pytest.raises(ValueError, match="nested too deeply"):
        load_run_record(path)
    with pytest.raises(ValueError, match="nested too deeply"):
        _json("{}")


@pytest.mark.parametrize("depth", [950, 5_000])
def test_parseable_but_deep_documents_do_not_escape_validation(
    tmp_path: Path, depth: int
) -> None:
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    recipe["environment"] = {**recipe["environment"], "x": "__DEEP__"}
    path = tmp_path / "recipe.json"
    path.write_text(_nest(json.dumps(recipe), depth), encoding="utf-8")
    try:
        load_study_recipe(path)
    except ValueError:
        pass  # rejected as invalid input is acceptable; a crash is not

    record = json.loads(RECORD.read_text(encoding="utf-8"))
    record["record"]["evidence"] = {"x": "__DEEP__"}
    path = tmp_path / "record.json"
    path.write_text(_nest(json.dumps(record), depth), encoding="utf-8")
    try:
        load_run_record(path)
    except ValueError:
        pass


@pytest.mark.parametrize(
    "argv",
    [
        ["recipe", "validate", "{path}"],
        ["inspect", "--json", "{path}"],
        ["compare", "--recipe", str(RECIPE), "{path}", str(RECORD)],
        ["verify", "--json", "--output", "{out}", "{path}"],
    ],
)
def test_cli_reports_deep_input_with_exit_two(tmp_path: Path, argv) -> None:
    path = tmp_path / "deep.json"
    path.write_text(TOO_DEEP, encoding="utf-8")
    args = [item.format(path=path, out=tmp_path / "verification") for item in argv]
    out, err = io.StringIO(), io.StringIO()
    assert main(args, stdout=out, stderr=err) == 2
    assert "Traceback" not in err.getvalue()


def test_fidelity_help_names_the_installed_command(capsys) -> None:
    out, err = io.StringIO(), io.StringIO()
    assert main(["fidelity", "--help"], stdout=out, stderr=err) == 0
    text = out.getvalue()
    assert "metria fidelity score --help" in text
    assert "kv-fidelity score" not in text
