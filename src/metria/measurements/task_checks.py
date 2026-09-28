"""Small deterministic workload checks, evaluated without retaining answer text."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from ..models import MetricDefinition, MetricDirection, MetricSample, MetricSummary
from ..recipes import _json_value, _unique_json_object

METHOD = "metria.task_checks"
VERSION = "1"
SCHEMA = "metria.task_checks.v1"
PASS_DEFINITION = MetricDefinition(
    "task_check_pass_rate",
    "fraction",
    MetricDirection.HIGHER_IS_BETTER,
    METHOD,
    VERSION,
)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _json_value(value, path="task_check"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def validate_checks(value: Any) -> tuple[dict[str, Any], ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) > 20
    ):
        raise ValueError("checks must be an array of at most 20 deterministic checks")
    checks = []
    seen = set()
    for item in value:
        if not isinstance(item, Mapping) or set(item) - {
            "id",
            "kind",
            "expected",
            "normalization",
            "required_keys",
        }:
            raise ValueError("invalid task-check fields")
        name, kind = item.get("id"), item.get("kind")
        if not isinstance(name, str) or not name.strip() or name in seen:
            raise ValueError(
                "task checks require unique nonempty IDs within each prompt"
            )
        seen.add(name)
        if kind not in {"exact_text", "nonempty", "json_object"}:
            raise ValueError("unknown deterministic task-check kind")
        check: dict[str, Any] = {"id": name, "kind": kind}
        if kind == "exact_text":
            expected = item.get("expected")
            normalization = item.get("normalization", "strip")
            if (
                not isinstance(expected, str)
                or len(expected) > 65536
                or normalization not in {"strip", "none"}
                or "required_keys" in item
            ):
                raise ValueError(
                    "exact_text needs bounded expected text and strip/none normalization"
                )
            check.update(expected=expected, normalization=normalization)
        elif kind == "json_object":
            keys = item.get("required_keys", ())
            if (
                isinstance(keys, (str, bytes))
                or not isinstance(keys, Sequence)
                or len(keys) > 20
                or any(not isinstance(key, str) or not key for key in keys)
                or len(set(keys)) != len(keys)
                or set(item) & {"expected", "normalization"}
            ):
                raise ValueError("json_object accepts only unique required_keys")
            check["required_keys"] = tuple(keys)
        elif set(item) != {"id", "kind"}:
            raise ValueError("nonempty accepts only id and kind")
        checks.append(check)
    return tuple(checks)


def prompt_checks(prompts: Any) -> tuple[tuple[dict[str, Any], ...], ...]:
    if isinstance(prompts, (str, bytes)) or not isinstance(prompts, Sequence):
        raise ValueError("task-check prompts must be an array")
    if any(not isinstance(row, Mapping) for row in prompts):
        raise ValueError("task-check prompt rows must be objects")
    return tuple(validate_checks(row.get("checks", ())) for row in prompts)


def _reject_constant(value: str) -> None:
    raise ValueError("nonfinite JSON constant")


def _passes(text: str, check: Mapping[str, Any]) -> bool:
    if check["kind"] == "nonempty":
        return bool(text.strip())
    if check["kind"] == "exact_text":
        expected = check["expected"]
        return (
            text.strip() == expected.strip()
            if check["normalization"] == "strip"
            else text == expected
        )
    try:
        value = json.loads(
            text, object_pairs_hook=_unique_json_object, parse_constant=_reject_constant
        )
    except (ValueError, TypeError, RecursionError):
        return False
    return isinstance(value, dict) and all(
        key in value for key in check["required_keys"]
    )


def evaluate_checks(
    text: str,
    checks: Sequence[Mapping[str, Any]],
    *,
    prompt_id: str,
    prompt_sha256: str,
) -> list[dict[str, Any]]:
    if not isinstance(text, str):
        raise ValueError("task checks require an observed output string")
    output_sha256 = hashlib.sha256(text.encode()).hexdigest()
    return [
        {
            "prompt_id": prompt_id,
            "prompt_sha256": prompt_sha256,
            "check_id": check["id"],
            "kind": check["kind"],
            "definition_sha256": _digest(check),
            "output_sha256": output_sha256,
            "passed": _passes(text, check),
        }
        for check in checks
    ]


def summarize_checks(
    rows: Sequence[Mapping[str, Any]],
    specs: Sequence[Sequence[Mapping[str, Any]]],
    prompt_rows: Sequence[Mapping[str, Any]],
    trials: int,
) -> tuple[dict[str, Any], MetricSummary | None]:
    declared = sum(len(checks) for checks in specs) * trials
    evidence = {
        "schema": SCHEMA,
        "method": METHOD,
        "method_version": VERSION,
        "workload_sha256": task_workload_identity(specs, prompt_rows, trials),
        "configured_prompts": sum(bool(checks) for checks in specs),
        "total_prompts": len(specs),
        "check_count": declared,
        "checks": tuple(rows),
    }
    if not declared:
        return {**evidence, "status": "not_configured"}, None
    if len(rows) != declared or any(
        not isinstance(row.get("passed"), bool) for row in rows
    ):
        raise ValueError("declared task-check evidence is incomplete")
    samples = tuple(
        MetricSample(
            float(row["passed"]),
            metadata={
                key: row[key] for key in ("prompt_id", "check_id", "definition_sha256")
            },
        )
        for row in rows
    )
    passed = sum(row["passed"] for row in rows)
    metric = MetricSummary(
        definition=PASS_DEFINITION,
        value=passed / declared,
        samples=samples,
        aggregation="mean",
        coverage=1.0,
    )
    return {**evidence, "status": "available", "passed": passed}, metric


def task_workload_identity(
    specs: Sequence[Sequence[Mapping[str, Any]]],
    prompt_rows: Sequence[Mapping[str, Any]],
    trials: int,
) -> str:
    identity = [
        {
            "id": row["id"],
            "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
            "definitions": [_digest(check) for check in checks],
        }
        for row, checks in zip(prompt_rows, specs, strict=True)
    ]
    return _digest({"prompts": identity, "trials": trials})
