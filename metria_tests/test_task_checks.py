import json

import pytest

from metria.measurements.task_checks import (
    evaluate_checks,
    prompt_checks,
    summarize_checks,
    validate_checks,
)


@pytest.mark.parametrize(
    "check,text,expected",
    [
        ({"id": "answer", "kind": "exact_text", "expected": "12"}, " 12\n", True),
        (
            {
                "id": "answer",
                "kind": "exact_text",
                "expected": "12",
                "normalization": "none",
            },
            " 12",
            False,
        ),
        ({"id": "answer", "kind": "nonempty"}, "   ", False),
        ({"id": "answer", "kind": "nonempty"}, "12", True),
        (
            {"id": "shape", "kind": "json_object", "required_keys": ["answer"]},
            '{"answer": 12}',
            True,
        ),
        (
            {"id": "shape", "kind": "json_object", "required_keys": ["answer"]},
            '{"other": 12}',
            False,
        ),
        ({"id": "shape", "kind": "json_object"}, '{"a": 1,"a": 2}', False),
        ({"id": "shape", "kind": "json_object"}, '{"a": NaN}', False),
        ({"id": "shape", "kind": "json_object"}, "[]", False),
    ],
)
def test_declared_checks_have_explicit_deterministic_semantics(check, text, expected):
    rows = evaluate_checks(
        text, validate_checks([check]), prompt_id="p1", prompt_sha256="a" * 64
    )
    assert rows[0]["passed"] is expected
    assert len(rows[0]["definition_sha256"]) == 64
    assert "expected" not in rows[0] and "output" not in rows[0]
    json.dumps(rows, allow_nan=False)


@pytest.mark.parametrize(
    "checks",
    [
        None,
        "bad",
        [{"id": "a", "kind": "unknown"}],
        [{"id": "a", "kind": "exact_text"}],
        [{"id": "a", "kind": "nonempty", "expected": "private"}],
        [{"id": "a", "kind": "json_object", "required_keys": ["a", "a"]}],
        [{"id": "a", "kind": "nonempty"}] * 2,
    ],
)
def test_invalid_checks_fail_before_inference(checks):
    with pytest.raises((ValueError, TypeError)):
        validate_checks(checks)


def test_check_evidence_retains_identity_without_private_expected_or_actual_text():
    spec = validate_checks(
        [{"id": "exact", "kind": "exact_text", "expected": "private expected answer"}]
    )
    rows = evaluate_checks(
        "private actual answer", spec, prompt_id="p1", prompt_sha256="a" * 64
    )
    assert "private" not in json.dumps(rows)
    evidence, metric = summarize_checks(
        rows, [spec], [{"id": "p1", "prompt": "private prompt"}], 1
    )
    assert evidence["passed"] == 0 and metric.value == 0
    assert metric.coverage == 1 and evidence["configured_prompts"] == 1
    assert "private" not in json.dumps(evidence)
    with pytest.raises(ValueError, match="incomplete"):
        summarize_checks([], [spec], [{"id": "p1", "prompt": "private prompt"}], 1)


def test_unconfigured_checks_are_not_a_zero_quality_score():
    prompts = [{"id": "p1", "prompt": "hello"}]
    evidence, metric = summarize_checks([], prompt_checks(prompts), prompts, 3)
    assert evidence["status"] == "not_configured" and metric is None
