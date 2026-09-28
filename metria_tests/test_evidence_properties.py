from __future__ import annotations

import json
import random
from dataclasses import replace

import pytest

from metria import (
    ComparisonPlan,
    MetricDefinition,
    MetricDirection,
    MetricSummary,
    RunRecord,
    RunSpec,
    RunStatus,
    compare_runs,
)
from metria.measurements import compare_trajectory_results
from metria.protocols import MeasurementResult
from metria.records import run_record_digest, run_record_to_json


def _record(model=None):
    return RunRecord(
        study_name="properties",
        run_id="one",
        requested=RunSpec(
            model=model or {"id": "model", "revision": "pinned"},
            runtime={"name": "fixture"},
            scenario={"seed": 1},
            measurements=("fixture",),
        ),
        resolved={},
        observed={},
        status=RunStatus.COMPLETED,
    )


def _capture(tokens):
    return MeasurementResult(
        evidence={
            "schema": "metria.trajectory_capture.v1",
            "method": "kv_fidelity.decode_time_trajectory",
            "method_version": "0.3.4",
            "n_prompts": 1,
            "prompts": [
                {
                    "id": "one",
                    "prompt_sha256": "a" * 64,
                    "token_ids": tokens,
                    "token_count": len(tokens),
                }
            ],
        }
    )


def test_digest_is_invariant_under_mapping_insertion_order():
    randomizer = random.Random(8128)
    expected = run_record_digest(_record())
    for _ in range(50):
        items = [("id", "model"), ("revision", "pinned")]
        randomizer.shuffle(items)
        record = _record(dict(items))
        assert run_record_digest(record) == expected
        assert (
            json.loads(run_record_to_json(record))["schema"] == "metria.run_record.v1"
        )


def test_trajectory_bounds_reflexivity_and_symmetry_over_generated_sequences():
    randomizer = random.Random(1729)
    for length in range(1, 50):
        reference = [randomizer.randrange(32) for _ in range(length)]
        candidate = reference[: randomizer.randrange(length + 1)] + [
            randomizer.randrange(32)
        ]
        forward = compare_trajectory_results(_capture(reference), _capture(candidate))
        reverse = compare_trajectory_results(_capture(candidate), _capture(reference))
        value = forward.metrics["trajectory_agreement_score"].value
        assert 0 <= value <= 100
        assert value == reverse.metrics["trajectory_agreement_score"].value
        assert (
            compare_trajectory_results(_capture(reference), _capture(reference))
            .metrics["trajectory_agreement_score"]
            .value
            == 100
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_metric_faults_cannot_enter_canonical_records(value):
    metric = MetricSummary(
        MetricDefinition(
            "value", "seconds", MetricDirection.LOWER_IS_BETTER, "fixture", "1"
        ),
        value,
    )
    with pytest.raises(ValueError):
        run_record_to_json(replace(_record(), metrics={"value": metric}))


def test_missing_identity_is_not_equality_or_a_valid_waiver():
    record = _record()
    dimension = "observed.identity.model.sha256"
    for plan in (
        ComparisonPlan(control=frozenset({dimension})),
        ComparisonPlan(waivers={dimension: "explicit test waiver"}),
    ):
        assert not compare_runs(record, record, plan).compatible


def test_comparison_roles_are_explicit_and_lifecycle_cannot_be_waived():
    reference = _record()
    candidate = _record({"id": "changed", "revision": "pinned"})
    assert compare_runs(
        reference, candidate, ComparisonPlan(vary=frozenset({"model.id"}))
    ).compatible
    assert not compare_runs(
        reference, candidate, ComparisonPlan(control=frozenset({"model.id"}))
    ).compatible
    failed = replace(reference, status=RunStatus.FAILED)
    assert not compare_runs(
        reference, failed, ComparisonPlan(vary=frozenset({"status"}))
    ).compatible
