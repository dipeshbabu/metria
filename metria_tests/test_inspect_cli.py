from __future__ import annotations

import io
import json
from pathlib import Path

from metria import (
    ComparisonPlan,
    RunSpec,
    StudyRecipe,
    StudySpec,
    TreatmentSpec,
    TreatmentType,
    dump_study_recipe,
)
from metria.cli import INSPECTION_SCHEMA, main


def _inspection_recipe(tmp_path: Path, *, override: bool = False) -> Path:
    trial_policy = (
        {"capability_overrides": ("turboquant.kv_cache.geometry",)} if override else {}
    )
    recipe = StudyRecipe(
        study=StudySpec(
            name="inspect-study",
            runs=(
                RunSpec(
                    model={
                        "id": "example/model",
                        "revision": "abc123",
                        "geometry": {
                            "hidden_size": 2048,
                            "num_attention_heads": 32,
                        },
                    },
                    runtime={"name": "llamacpp"},
                    scenario={"max_tokens": 8},
                    measurements=("example.measurement",),
                    treatments=(
                        TreatmentSpec(
                            name="llamacpp.kv_cache",
                            kind=TreatmentType.RUNTIME_FEATURE,
                            config={
                                "key_dtype": "q8_0",
                                "value_dtype": "turbo3",
                            },
                        ),
                    ),
                    trial_policy=trial_policy,
                ),
            ),
            comparison=ComparisonPlan(),
        ),
        measurement_configs={"example.measurement": {}},
        environment={},
    )
    path = tmp_path / "inspect.json"
    dump_study_recipe(path, recipe)
    return path


def test_inspect_json_reports_fail_closed_capabilities(tmp_path: Path) -> None:
    path = _inspection_recipe(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()

    status = main(
        ["inspect", str(path), "--json"],
        stdout=stdout,
        stderr=stderr,
    )
    payload = json.loads(stdout.getvalue())

    assert status == 0
    assert stderr.getvalue() == ""
    assert payload["schema"] == INSPECTION_SCHEMA
    assert payload["study"] == "inspect-study"
    assert payload["hardware"]["host"].get("hostname") is None
    assert len(payload["hardware"]["host"]["hostname_sha256"]) == 64
    run = payload["runs"][0]
    assert run["geometry"]["head_dim"] == 64
    assert run["allowed"] is False
    assert run["blocking"] == ["turboquant.kv_cache.geometry"]
    assert (
        run["capabilities"]["turboquant.kv_cache.geometry"]["status"] == "unsupported"
    )


def test_inspect_json_records_explicit_experimental_override(tmp_path: Path) -> None:
    path = _inspection_recipe(tmp_path, override=True)
    stdout = io.StringIO()

    status = main(
        ["inspect", str(path), "--json"],
        stdout=stdout,
        stderr=io.StringIO(),
    )
    payload = json.loads(stdout.getvalue())
    capability = payload["runs"][0]["capabilities"]["turboquant.kv_cache.geometry"]

    assert status == 0
    assert payload["runs"][0]["allowed"] is True
    assert capability["status"] == "experimental"
    assert capability["evidence"]["experimental_override"] is True


def test_inspect_human_output_is_concise(tmp_path: Path) -> None:
    path = _inspection_recipe(tmp_path)
    stdout = io.StringIO()

    status = main(
        ["inspect", str(path)],
        stdout=stdout,
        stderr=io.StringIO(),
    )

    assert status == 0
    text = stdout.getvalue()
    assert "study inspect-study" in text
    assert "runtime=llamacpp allowed=false" in text
    assert "turboquant.kv_cache.geometry: unsupported" in text
    assert "hidden_size" not in text
