from __future__ import annotations

from metria import ComparisonPlan


def test_comparison_plan_preserves_legacy_positional_analyses_argument() -> None:
    plan = ComparisonPlan(
        frozenset({"runtime"}),
        frozenset({"model"}),
        frozenset({"observed.hardware_class"}),
        ("kv_fidelity.trajectory_match",),
    )

    assert plan.vary == frozenset({"runtime"})
    assert plan.control == frozenset({"model"})
    assert plan.block_by == frozenset({"observed.hardware_class"})
    assert plan.analyses == ("kv_fidelity.trajectory_match",)
    assert dict(plan.waivers) == {}
