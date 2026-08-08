from __future__ import annotations

import io
import json
from pathlib import Path

from metria import (
    ComparisonPlan,
    MetricDefinition,
    MetricDirection,
    MetricSummary,
    RunRecord,
    RunSpec,
    RunStatus,
    StudyRecipe,
    StudySpec,
    dump_run_record,
    dump_study_recipe,
)
from metria.cli import COMPARISON_REPORT_SCHEMA, main


def _run(runtime: str) -> RunSpec:
    return RunSpec(
        model={"id": "example/model", "revision": "abc123"},
        runtime={"name": runtime},
        scenario={"name": "decode", "max_tokens": 8},
        measurements=("latency",),
    )


def _recipe(tmp_path: Path) -> tuple[Path, RunSpec, RunSpec]:
    left = _run("llamacpp")
    right = _run("vllm")
    recipe = StudyRecipe(
        study=StudySpec(
            name="compare-study",
            runs=(left, right),
            comparison=ComparisonPlan(
                vary=frozenset({"runtime"}),
                control=frozenset({"model", "scenario", "measurements"}),
            ),
        ),
        measurement_configs={"latency": {}},
        environment={},
    )
    path = tmp_path / "study.json"
    dump_study_recipe(path, recipe)
    return path, left, right


def _record(
    spec: RunSpec,
    run_id: str,
    *,
    method: str = "wall_clock",
) -> RunRecord:
    return RunRecord(
        study_name="compare-study",
        run_id=run_id,
        requested=spec,
        resolved={"runtime": dict(spec.runtime)},
        observed={"runtime": spec.runtime["name"], "hardware_class": "same"},
        status=RunStatus.COMPLETED,
        metrics={
            "latency_ms": MetricSummary(
                definition=MetricDefinition(
                    name="latency_ms",
                    unit="ms",
                    direction=MetricDirection.LOWER_IS_BETTER,
                    method=method,
                    version="1",
                ),
                value=10.0,
            )
        },
    )


def test_compare_cli_uses_explicit_recipe_plan(tmp_path: Path) -> None:
    recipe_path, left_spec, right_spec = _recipe(tmp_path)
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    dump_run_record(left_path, _record(left_spec, "left"))
    dump_run_record(right_path, _record(right_spec, "right"))
    stdout = io.StringIO()
    stderr = io.StringIO()

    status = main(
        [
            "compare",
            str(left_path),
            str(right_path),
            "--recipe",
            str(recipe_path),
            "--json",
        ],
        stdout=stdout,
        stderr=stderr,
    )
    payload = json.loads(stdout.getvalue())

    assert status == 0
    assert stderr.getvalue() == ""
    assert payload["schema"] == COMPARISON_REPORT_SCHEMA
    assert payload["compatible"] is True
    assert payload["pairs"][0]["report"]["comparable_metrics"] == ["latency_ms"]
    assert len(payload["records"][0]["record_digest"]) == 64
    assert len(payload["records"][0]["evidence_digest"]) == 64


def test_compare_cli_reports_metric_method_mismatch_as_incompatible(tmp_path: Path) -> None:
    recipe_path, left_spec, right_spec = _recipe(tmp_path)
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    dump_run_record(left_path, _record(left_spec, "left", method="wall_clock"))
    dump_run_record(right_path, _record(right_spec, "right", method="estimated"))
    stdout = io.StringIO()

    status = main(
        ["compare", str(left_path), str(right_path), "--recipe", str(recipe_path)],
        stdout=stdout,
        stderr=io.StringIO(),
    )

    assert status == 1
    assert "compatible=false" in stdout.getvalue()
    assert "metric latency_ms" in stdout.getvalue()


def test_compare_cli_rejects_record_not_bound_to_recipe(tmp_path: Path) -> None:
    recipe_path, left_spec, _ = _recipe(tmp_path)
    left_path = tmp_path / "left.json"
    foreign = _run("sglang")
    dump_run_record(left_path, _record(left_spec, "left"))
    foreign_path = tmp_path / "foreign.json"
    dump_run_record(foreign_path, _record(foreign, "foreign"))
    stderr = io.StringIO()

    status = main(
        [
            "compare",
            str(left_path),
            str(foreign_path),
            "--recipe",
            str(recipe_path),
        ],
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert status == 2
    assert "requested RunSpec is not present" in stderr.getvalue()


def test_compare_cli_requires_two_records(tmp_path: Path) -> None:
    recipe_path, left_spec, _ = _recipe(tmp_path)
    left_path = tmp_path / "left.json"
    dump_run_record(left_path, _record(left_spec, "left"))
    stderr = io.StringIO()

    status = main(
        ["compare", str(left_path), "--recipe", str(recipe_path)],
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert status == 2
    assert "requires at least two" in stderr.getvalue()
