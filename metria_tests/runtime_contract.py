"""Reusable semantic contract checks for Metria runtime adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from metria import RunSpec, SupportLevel
from metria.protocols import (
    CaptureRequest,
    InferenceBatch,
    InferenceRequest,
    RuntimeAdapter,
    RuntimeSession,
    SupportReport,
)


@dataclass(frozen=True)
class RuntimeContractCase:
    """Inputs and expectations for one adapter conformance exercise."""

    adapter: RuntimeAdapter
    spec: RunSpec
    environment: Mapping[str, Any] = field(default_factory=dict)
    requests: tuple[InferenceRequest, ...] = (
        InferenceRequest(prompt="metria contract prompt"),
    )
    capture: tuple[CaptureRequest, ...] = ()
    privacy_terms: tuple[str, ...] = ("metria contract prompt",)


def _assert_no_sensitive_text(value: object, terms: Sequence[str]) -> None:
    rendered = repr(value)
    for term in terms:
        assert term not in rendered, f"runtime evidence leaked sensitive text: {term!r}"


def _assert_supported(report: SupportReport) -> None:
    assert isinstance(report, SupportReport)
    assert report.status in {SupportLevel.SUPPORTED, SupportLevel.EXPERIMENTAL}
    assert all(isinstance(reason, str) and reason.strip() for reason in report.reasons)


def exercise_runtime_contract(case: RuntimeContractCase) -> None:
    """Exercise the lifecycle/evidence contract shared by supported adapters.

    Engine-specific semantics such as CLI flags, tokenizer behavior, kernel
    selection, and specialized captures remain in adapter-specific tests. This
    helper checks only invariants Metria expects from every supported runtime.
    """

    adapter = case.adapter
    assert isinstance(adapter.name, str) and adapter.name.strip()

    support = adapter.probe(case.spec, case.environment)
    _assert_supported(support)
    _assert_no_sensitive_text(support.evidence, case.privacy_terms)

    first_resolved = adapter.resolve(case.spec, case.environment)
    second_resolved = adapter.resolve(case.spec, case.environment)
    assert first_resolved == second_resolved
    assert isinstance(first_resolved, Mapping)
    _assert_no_sensitive_text(first_resolved, case.privacy_terms)

    session: RuntimeSession = adapter.launch(first_resolved, case.environment)
    try:
        batch = session.infer(case.requests, case.capture)
        assert isinstance(batch, InferenceBatch)
        assert len(batch.outputs) == len(case.requests)
        _assert_no_sensitive_text(batch.metadata, case.privacy_terms)

        observed_before_reset = adapter.observe(session)
        assert isinstance(observed_before_reset, Mapping)
        assert observed_before_reset.get("runtime") == adapter.name
        _assert_no_sensitive_text(observed_before_reset, case.privacy_terms)

        session.reset("contract")
        observed_after_reset = adapter.observe(session)
        assert isinstance(observed_after_reset, Mapping)
        assert observed_after_reset.get("runtime") == adapter.name
        _assert_no_sensitive_text(observed_after_reset, case.privacy_terms)
    finally:
        session.close()

    # Cleanup should be safe to call again, and a closed session must never
    # silently execute another inference request.
    session.close()
    with pytest.raises(RuntimeError):
        session.infer(case.requests, case.capture)
