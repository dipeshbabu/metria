"""Immutable baselines: capture, integrity, freshness and per-metric eligibility."""

from __future__ import annotations

import io
import json
import shutil
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from metria import baseline
from metria._demo_worker import _run_fixture_in
from metria.cli import main
from metria.recipes import dump_study_recipe, load_study_recipe
from metria.verification import render_verification

NOW = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
DAY = timedelta(days=1)


def _bundle(tmp_path: Path, case: str = "pass", *, synthetic: bool = False):
    root = tmp_path / f"fixture-{case}"
    root.mkdir()
    output = tmp_path / f"bundle-{case}"
    _run_fixture_in(root, case, output)
    data = json.loads((output / "verification.json").read_text(encoding="utf-8"))
    if not synthetic:
        # Present the fixture as non-synthetic so eligibility can be exercised;
        # the bundle stays internally consistent.
        data.pop("fixture_only", None)
        text = json.dumps(data, indent=2, sort_keys=True) + "\n"
        (output / "verification.json").write_text(text, encoding="utf-8")
        (output / "manifest.json").write_text(text, encoding="utf-8")
        (output / "report.md").write_text(render_verification(data), encoding="utf-8")
    return output, root / "study.json", data


@pytest.fixture
def captured(tmp_path):
    evidence, recipe, data = _bundle(tmp_path)
    store = tmp_path / "store"
    receipt = baseline.capture(
        evidence,
        recipe,
        store,
        max_age=7 * DAY,
        performance_max_age=DAY,
        now=NOW,
    )
    return store / receipt["baseline_id"], recipe, data, receipt


def _check(path, recipe, data, now=NOW + timedelta(hours=1), hardware=None):
    return baseline.check(
        path, recipe, now=now, hardware=hardware if hardware else data["hardware"]
    )


def test_fresh_baseline_is_reusable_for_identical_controls(captured):
    path, recipe, data, receipt = captured
    assert path.name == receipt["baseline_id"]
    assert baseline.validate(path) == receipt
    assert sorted(item.name for item in path.iterdir()) == sorted(
        [*baseline.BUNDLE_FILES, "recipe.json", "receipt.json"]
    )
    result = _check(path, recipe, data)
    assert result["decision"] == "reuse_reference"
    assert result["reusable"] == list(baseline.FAMILIES)
    assert result["fresh_paired_run_required_for"] == []


def test_stale_baseline_never_passes(captured):
    path, recipe, data, receipt = captured
    result = _check(path, recipe, data, now=NOW + 8 * DAY)
    assert result["decision"] == "fresh_paired_run"
    assert all(not row["eligible"] for row in result["families"].values())
    assert any("expired" in r for r in result["families"]["behavior"]["reasons"])
    early = _check(path, recipe, data, now=NOW - DAY)
    assert (
        "baseline was created in the future" in early["families"]["quality"]["reasons"]
    )


def test_performance_reuse_needs_window_and_identical_hardware(captured):
    path, recipe, data, _ = captured
    later = _check(path, recipe, data, now=NOW + 2 * DAY)
    assert later["reusable"] == ["behavior", "quality"]
    assert "older than its reuse window" in later["families"]["timing"]["reasons"][0]
    other = _check(path, recipe, data, hardware={"platform": {"system": "other"}})
    assert other["families"]["memory"]["reasons"] == [
        "hardware differs from the baseline"
    ]


def test_default_capture_never_reuses_performance(tmp_path):
    evidence, recipe, data = _bundle(tmp_path)
    receipt = baseline.capture(evidence, recipe, tmp_path / "s", max_age=DAY, now=NOW)
    result = _check(tmp_path / "s" / receipt["baseline_id"], recipe, data)
    assert result["reusable"] == ["behavior", "quality"]
    assert result["families"]["timing"]["reasons"] == [
        "no performance reuse window was set at capture"
    ]


def _changed(recipe_path: Path, tmp_path: Path, change) -> Path:
    recipe = load_study_recipe(recipe_path)
    target = tmp_path / "changed.json"
    dump_study_recipe(target, change(recipe))
    return target


def test_changed_controls_invalidate_with_reasons(captured, tmp_path):
    path, recipe_path, data, _ = captured

    def workload(recipe):
        configs = {
            name: {**config, "prompts": [{"id": "x", "prompt": "different"}]}
            for name, config in recipe.measurement_configs.items()
        }
        return replace(recipe, measurement_configs=configs)

    def runtime(recipe):
        first = recipe.study.runs[0]
        moved = replace(first, runtime={**first.runtime, "threads": 3})
        study = replace(recipe.study, runs=(moved, recipe.study.runs[1]))
        return replace(recipe, study=study)

    def model(recipe):
        first = recipe.study.runs[0]
        moved = replace(first, model={**first.model, "sha256": "0" * 64})
        study = replace(recipe.study, runs=(moved, recipe.study.runs[1]))
        return replace(recipe, study=study)

    def environment(recipe):
        return replace(
            recipe, environment={"llama_cpp_token_ids_capture_sha256": "1" * 64}
        )

    expected = {
        workload: "workload or task checks changed",
        runtime: "runtime, deployment or trust configuration changed",
        model: "model or tokenizer changed",
        environment: "capture environment changed",
    }
    for change, reason in expected.items():
        result = _check(path, _changed(recipe_path, tmp_path, change), data)
        assert result["decision"] == "fresh_paired_run"
        assert reason in result["families"]["behavior"]["reasons"], result


def test_methodology_and_synthetic_reasons(tmp_path, monkeypatch):
    evidence, recipe, data = _bundle(tmp_path, synthetic=True)
    with pytest.raises(baseline.BaselineError, match="allow-synthetic"):
        baseline.capture(evidence, recipe, tmp_path / "s", max_age=DAY, now=NOW)
    receipt = baseline.capture(
        evidence, recipe, tmp_path / "s", max_age=DAY, allow_synthetic=True, now=NOW
    )
    path = tmp_path / "s" / receipt["baseline_id"]
    reasons = _check(path, recipe, data)["families"]["behavior"]["reasons"]
    assert reasons == ["baseline is synthetic demo evidence"]
    monkeypatch.setattr("metria.__version__", "9.9.9")
    reasons = _check(path, recipe, data)["families"]["behavior"]["reasons"]
    assert "verifier implementation changed" in reasons


@pytest.mark.parametrize(
    "tamper",
    [
        lambda p: (p / "reference.run.json").write_text(
            (p / "reference.run.json").read_text().replace("run-0000", "run-9999")
        ),
        lambda p: (p / "receipt.json").write_text(
            (p / "receipt.json").read_text().replace("2026-11-08", "2099-11-08")
        ),
        lambda p: (p / "recipe.json").write_text((p / "recipe.json").read_text() + " "),
        lambda p: (p / "report.md").write_text("edited"),
        lambda p: (p / "candidate.run.json").unlink(),
        lambda p: (p / "extra.txt").write_text("x"),
        lambda p: (p / "receipt.json").write_text("{not json"),
        lambda p: (p / "receipt.json").write_text('{"schema": "other"}'),
    ],
)
def test_tampered_or_partial_baselines_fail_closed(captured, tamper):
    path, recipe, data, _ = captured
    tamper(path)
    with pytest.raises(baseline.BaselineError):
        baseline.validate(path)
    with pytest.raises(baseline.BaselineError):
        _check(path, recipe, data)


def test_renamed_or_missing_directory_fails(captured, tmp_path):
    path, *_ = captured
    renamed = path.with_name("f" * 64)
    shutil.copytree(path, renamed)
    with pytest.raises(baseline.BaselineError, match="altered"):
        baseline.validate(renamed)
    with pytest.raises(baseline.BaselineError, match="not a baseline"):
        baseline.validate(tmp_path / "missing")


def test_duplicates_and_concurrent_writes_never_overwrite(tmp_path, monkeypatch):
    evidence, recipe, _ = _bundle(tmp_path)
    store = tmp_path / "store"
    first = baseline.capture(evidence, recipe, store, max_age=DAY, now=NOW)
    with pytest.raises(baseline.BaselineError, match="already baseline"):
        baseline.capture(evidence, recipe, store, max_age=DAY, now=NOW + DAY)
    # A racing writer that published the same ID first must not be replaced.
    monkeypatch.setattr(baseline, "_receipts", lambda root: [])
    original = (store / first["baseline_id"] / "receipt.json").read_bytes()
    with pytest.raises(baseline.BaselineError, match="could not create"):
        baseline.capture(evidence, recipe, store, max_age=DAY, now=NOW)
    assert (store / first["baseline_id"] / "receipt.json").read_bytes() == original
    assert [p.name for p in store.iterdir()] == [first["baseline_id"]]


def test_capture_refuses_unusable_evidence(tmp_path):
    evidence, recipe, _ = _bundle(tmp_path, "not-comparable")
    with pytest.raises(baseline.BaselineError, match="cannot become a baseline"):
        baseline.capture(evidence, recipe, tmp_path / "s", max_age=DAY)
    good, good_recipe, _ = _bundle(tmp_path)
    with pytest.raises(baseline.BaselineError, match="not the one"):
        baseline.capture(good, recipe, tmp_path / "s", max_age=DAY)
    with pytest.raises(baseline.BaselineError, match="max age must be positive"):
        baseline.capture(good, good_recipe, tmp_path / "s", max_age=timedelta(0))
    with pytest.raises(baseline.BaselineError, match="performance max age"):
        baseline.capture(
            good, good_recipe, tmp_path / "s", max_age=DAY, performance_max_age=2 * DAY
        )
    (good / "report.md").write_text("edited")
    with pytest.raises(baseline.BaselineError, match="bundle is invalid"):
        baseline.capture(good, good_recipe, tmp_path / "s", max_age=DAY)


def test_reference_gaps_are_refused():
    manifest = {
        "verdict": "PASS",
        "records": {"reference": {"status": "completed", "evidence_gaps": ["x"]}},
    }
    with pytest.raises(baseline.BaselineError, match="evidence gaps"):
        baseline._check_reusable(manifest, False)


def test_timestamps_are_validated():
    with pytest.raises(baseline.BaselineError, match="must be a timestamp"):
        baseline._parse_time(None, "created_at")
    with pytest.raises(baseline.BaselineError, match="not an ISO"):
        baseline._parse_time("yesterday", "created_at")
    with pytest.raises(baseline.BaselineError, match="timezone"):
        baseline._parse_time("2026-01-01T00:00:00", "created_at")


def _cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    code = main([str(item) for item in argv], stdout=out, stderr=err)
    return code, out.getvalue() + err.getvalue()


def test_cli_round_trip(tmp_path):
    evidence, recipe, _ = _bundle(tmp_path)
    store = tmp_path / "store"
    code, output = _cli(
        "baseline",
        "capture",
        "--evidence",
        evidence,
        "--recipe",
        recipe,
        "--store",
        store,
        "--max-age-days",
        7,
        "--performance-max-age-hours",
        12,
    )
    assert code == 0, output
    baseline_id = output.split()[2]
    path = store / baseline_id
    assert _cli("baseline", "validate", path) == (
        0,
        f"Baseline {baseline_id} is intact\n",
    )
    code, output = _cli("baseline", "inspect", path)
    assert code == 0 and json.loads(output)["baseline_id"] == baseline_id
    code, output = _cli("baseline", "check", path, "--recipe", recipe, "--json")
    result = json.loads(output)
    # Host hardware differs from the synthetic fixture, so performance is fresh.
    assert code == 0 and result["reusable"] == ["behavior", "quality"]
    code, output = _cli("baseline", "check", path, "--recipe", recipe)
    assert code == 0 and "timing: fresh run" in output and "hardware differs" in output
    (path / "report.md").write_text("edited")
    code, output = _cli("baseline", "check", path, "--recipe", recipe)
    assert code == 2 and "metria: error" in output


def test_cli_check_exit_code_for_fresh_run(tmp_path):
    evidence, recipe, _ = _bundle(tmp_path, synthetic=True)
    store = tmp_path / "store"
    code, output = _cli(
        "baseline",
        "capture",
        "--evidence",
        evidence,
        "--recipe",
        recipe,
        "--store",
        store,
        "--max-age-days",
        1,
        "--allow-synthetic",
    )
    assert code == 0, output
    path = store / output.split()[2]
    code, output = _cli("baseline", "check", path, "--recipe", recipe)
    assert code == 3 and "fresh_paired_run" in output
