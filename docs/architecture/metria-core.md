# Metria core architecture

Metria is a neutral evidence and experiment layer for LLM inference systems.
Its job is not to reimplement serving engines, quantizers, kernels, or schedulers.
Instead, Metria defines how an inference-systems study is specified, executed,
recorded, and compared.

The initial design deliberately keeps the public model small:

```text
StudySpec
  ├── runs: RunSpec[]
  └── comparison: ComparisonPlan

RunSpec
  ├── model
  ├── runtime
  ├── treatments
  ├── scenario
  ├── measurements
  ├── trial policy
  └── environment selector

MeasurementResult
  ├── metrics
  ├── evidence
  └── artifacts

RunRecord
  ├── requested
  ├── resolved
  ├── observed
  ├── status
  ├── metrics
  ├── events
  ├── artifacts
  └── provenance
```

## Study semantics

The study, rather than a hard-coded fingerprint, decides what may vary.
A comparison plan separates:

- `vary`: dimensions intentionally changed by the study;
- `control`: dimensions that must match for a valid comparison;
- `block_by`: dimensions used to form comparable groups.

For example, a study may compare two runtimes and two KV-cache treatments while
controlling the model, scenario, and trial policy and blocking by hardware
class. Runtime therefore is not globally "compatible" or "incompatible"; its
role is study-specific.

## Requested, resolved, observed

Metria keeps three states separate:

1. **Requested**: what the recipe asked for.
2. **Resolved**: exact revisions, runtime settings, artifacts, and choices
   selected before execution.
3. **Observed**: what the runtime and environment report actually ran.

This distinction is required for trustworthy systems evidence. A request for
FP8 KV cache, for example, is not evidence that the launched engine used FP8.
Runtime adapters must eventually provide applied-configuration evidence in the
observed record.

## Treatments

`TreatmentSpec` is intentionally broader than "optimization". The initial
taxonomy is:

- model transformation;
- runtime feature;
- execution policy;
- instrumentation.

The common study model treats these as experimental treatments while leaving
their different lifecycle mechanics to specialized adapters.

## Metrics

A metric identity includes its name, unit, optimization direction, method, and
method version. Raw values are directly comparable only when those identities
match and the study comparison plan permits the run comparison.

This prevents methodologically different measurements, such as full-vocabulary
KL divergence and top-k KL estimates, from being silently treated as the same
metric. Cross-method studies can still compare matched-baseline effect sizes,
but that analysis must be explicit.

## Measurement results and evidence

`MeasurementProtocol` returns a `MeasurementResult`, not only a metric mapping.
The result separates compact numerical summaries from the observations needed
to reproduce or derive them:

- `metrics` contains typed `MetricSummary` values;
- `evidence` contains immutable method-specific observations;
- `artifacts` points to large external payloads when embedding them would make a
  run record impractical.

This distinction is important for fidelity methods. A decode-time token
trajectory, for example, is evidence produced by one run. Its agreement score
is not a property of that run by itself; the score exists only after comparing a
reference trajectory with a candidate trajectory. Metria therefore retains the
run-local trajectories first and derives the pairwise score at comparison time.

Pairwise derived metrics must verify that the underlying evidence refers to the
same experimental object. The trajectory bridge checks prompt identifiers,
prompt fingerprints, capture schema, and method version before computing a
score. This avoids silently comparing two different prompt sets that happen to
share a benchmark label.

## Runtime and measurement boundaries

The provisional protocols separate runtime lifecycle from measurement
methodology:

- `RuntimeAdapter` probes support, resolves configuration, launches a session,
  and records observed runtime evidence.
- `RuntimeSession` performs inference and owns reset/cleanup behavior.
- `MeasurementProtocol` declares evidence requirements and returns a
  `MeasurementResult` containing metrics and retained evidence.

These protocols are intentionally provisional until exercised by at least two
materially different runtimes.

## What is not in the first core

The first Metria core does not provide:

- automated configuration recommendation;
- active search;
- a universal optimization plugin;
- a universal fidelity scalar;
- production serving;
- production TurboQuant integration;
- a root all-runtime dependency bundle.

The first stable milestone is narrower:

> Metria can reproducibly run, record, and validly compare a defined
> inference-systems study across at least two runtimes.

KV Fidelity and TurboQuant Reference remain independent focused components while
Metria's shared study and evidence model matures.
