# Metria pairwise analyses

Metria separates **comparison validity** from **derived pairwise analysis**.

`compare_runs()` first decides whether two completed run records are directly
comparable under the study's `ComparisonPlan`. Only after that gate passes can a
registered pairwise analyzer derive metrics from the two run records.

This avoids three common mistakes:

1. computing a fidelity/effect metric for runs that violate the study controls;
2. silently accepting an undeclared model, tokenizer, runtime, generation, or
   applied-configuration difference;
3. hiding an analysis failure by treating it as if no analysis had been
   requested.

## Fail-closed comparison

Comparison is closed-world with respect to differences.

A comparison-relevant difference is valid only when it is covered by one of
these roles:

- `vary`: the difference is an intentional treatment/change;
- `control`: the value must match;
- `block_by`: the value must match within a direct comparison block;
- `waivers`: the difference may be ignored only with an explicit retained
  reason.

A difference that is not covered by one of these roles makes the pair
incompatible. Missing required evidence is distinct from an explicit `null`
value and fails closed.

For example:

```python
from metria import ComparisonPlan

plan = ComparisonPlan(
    vary=frozenset(
        {
            "treatments",
            "resolved.kv_cache",
            "observed.configured.kv_cache",
            "observed.applied.fields.cache.cache_dtype",
        }
    ),
    control=frozenset(
        {
            "model",
            "runtime",
            "scenario",
            "measurements",
            "trial_policy",
        }
    ),
    block_by=frozenset({"observed.hardware_class"}),
    analyses=("kv_fidelity.trajectory_match",),
)
```

This says that the KV-cache treatment and the corresponding resolved,
configured, and observed applied state are intentional differences. Model,
runtime, workload/generation request, and measurement identity stay controlled.

## Nested evidence paths

Comparison paths may address nested requested, resolved, and observed evidence.

Examples include:

```text
model.revision
model.tokenizer_revision
runtime.version
scenario.temperature
resolved.model.revision
resolved.runtime.version
resolved.kv_cache.dtype
observed.runtime.version
observed.configured.kv_cache.dtype
observed.applied.fields.cache.cache_dtype
observed.hardware.accelerator
```

Runtime introspection sometimes stores a key containing dots, such as
`"cache.cache_dtype"`. Metria resolves the natural dotted comparison path
without requiring callers to know that storage detail.

A top-level requested declaration for `model`, `runtime`, or `scenario` also
governs the corresponding direct resolved/observed lifecycle evidence unless a
more specific resolved/observed declaration is present. Treatment-specific
downstream evidence is intentionally not guessed. If a KV-cache setting is the
change under test, declare the requested treatment and the relevant
resolved/observed KV-cache paths explicitly.

## Default comparison-relevant evidence

Metria examines all requested run dimensions:

```text
model
runtime
scenario
measurements
treatments
trial_policy
environment_selector
```

It also examines resolved and observed evidence by default. Known bookkeeping
that is not configuration identity, such as cleanup state and reset counts, is
excluded.

Invocation evidence is projected down to comparison-relevant identity fields
rather than comparing output-dependent values. The current projection includes
prompt fingerprints, rendered-prompt fingerprints, system-prompt fingerprints,
and generation settings. Output token count and request duration do not make a
pair invalid merely because the model behaved differently.

Runtime probe/support diagnostics under `resolved.support` are also not treated
as configuration identity. Support diagnostics may differ without changing the
system under test.

## Waivers

A waiver is an explicit exception with a required human-readable reason:

```python
plan = ComparisonPlan(
    waivers={
        "observed.environment.zone": "cross-zone qualification",
    }
)
```

When the waived value differs, the reason is retained in
`CompatibilityReport.waived_differences` and in machine-readable `metria
compare` output.

Waivers are deliberately narrow:

- the path must resolve on both runs;
- a missing waived value is still insufficient evidence;
- waiver paths cannot overlap a `vary`, `control`, or `block_by` subtree;
- a waiver never turns method-incompatible raw metrics into compatible metrics.

Use a waiver only when the difference is understood and intentionally accepted.
Do not use it to compensate for missing provenance.

## Declaring analyses

Pairwise analyses are part of the comparison plan:

```python
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
The planned `metria verify` workflow will make reference/candidate roles a
first-class user concept, but basic comparison does not infer a baseline from
runtime names or treatment labels.

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

- analyses spanning more than two runs;
- statistical pooling across repeated trials;
- automatic analyzer selection from evidence types;
- a universal comparability fingerprint;
- Pareto/recommendation logic.

Those should build on retained run and pair evidence rather than being folded
into basic compatibility semantics.
