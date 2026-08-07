# Metria pairwise analyses

Metria separates **comparison validity** from **derived pairwise analysis**.

`compare_runs()` first decides whether two completed run records are directly
comparable under the study's `ComparisonPlan`. Only after that gate passes can a
registered pairwise analyzer derive metrics from the two run records.

This avoids two common mistakes:

1. computing a fidelity/effect metric for runs that violate the study controls;
2. hiding an analysis failure by treating it as if no analysis had been
   requested.

## Declaring analyses

Pairwise analyses are part of the comparison plan:

```python
from metria import ComparisonPlan

plan = ComparisonPlan(
    vary=frozenset({"runtime"}),
    control=frozenset({"model", "scenario", "measurements"}),
    block_by=frozenset({"observed.hardware_class"}),
    analyses=("kv_fidelity.trajectory_match",),
)
```

Names are ordered and unique. The order is preserved in each pair's retained
analysis outcomes.

## Analyzer contract

A pairwise analyzer is a small named/versioned object:

```python
class PairwiseAnalysis:
    name: str
    version: str

    def analyze(left: RunRecord, right: RunRecord) -> MeasurementResult:
        ...
```

The registry key must equal `analysis.name`. Every analysis named by the study
must be registered before execution begins; otherwise `execute_study()` fails
before the first run starts.

## Pair lifecycle

For each deterministic run pair, Metria first computes a
`CompatibilityReport`.

Then every requested analyzer receives one of three outcomes:

- `completed`: the pair was compatible and the analyzer returned a
  `MeasurementResult`;
- `skipped`: the pair was not directly comparable, so analyzer code was not
  called;
- `failed`: the pair was compatible but analyzer code raised an exception.

Analyzer failure does not abort the study. Metria records the exception type and
a SHA-256 fingerprint of the message instead of retaining arbitrary raw error
text.

## Left/right ordering

The initial executor uses deterministic study order: the earlier `RunSpec`
becomes `left` and the later run becomes `right`.

This matters for directional analyses. A study that interprets the left side as
a baseline/reference should therefore place the baseline earlier in the study.
Future explicit reference/candidate selectors may replace this convention, but
Metria does not infer a baseline from runtime names or treatment labels today.

## Built-in trajectory agreement

`TrajectoryAgreementAnalysis` is the first built-in pairwise analyzer:

```python
from metria.measurements import (
    TokenTrajectoryProtocol,
    TrajectoryAgreementAnalysis,
)

measurement = TokenTrajectoryProtocol()
analysis = TrajectoryAgreementAnalysis()
```

The measurement captures run-local decode-time token trajectories. The analyzer
recovers those retained captures from each `RunRecord` and delegates to the same
`compare_trajectory_results()` implementation used for direct pairwise scoring.
The KV Fidelity v0.3.4 scoring rule therefore remains single-sourced.

A complete study can declare it as:

```python
study = StudySpec(
    name="runtime-trajectory-study",
    runs=(reference_run, candidate_run),
    comparison=ComparisonPlan(
        vary=frozenset({"runtime"}),
        control=frozenset({"model", "scenario", "measurements"}),
        analyses=(analysis.name,),
    ),
)

result = execute_study(
    study,
    adapters={...},
    measurements={measurement.name: measurement},
    measurement_configs={measurement.name: {...}},
    environment={...},
    analyses={analysis.name: analysis},
)
```

The derived trajectory score then lives under the pair's completed analysis
outcome, not on either individual run.

## Current limits

The first pairwise-analysis layer intentionally does not yet provide:

- explicit baseline/reference selectors beyond study order;
- analyses spanning more than two runs;
- statistical pooling across repeated trials;
- automatic analyzer selection from evidence types;
- persistence or report rendering;
- Pareto/recommendation logic.

Those should build on retained run and pair evidence rather than being folded
into basic compatibility semantics.
