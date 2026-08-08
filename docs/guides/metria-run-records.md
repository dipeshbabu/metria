# Metria run records and comparison

Metria persists executed run evidence with the versioned schema:

```text
metria.run_record.v1
```

A run record is deliberately different from a study recipe. A recipe describes
**requested intent**. A run record retains requested intent together with the
resolved configuration, observed runtime state, typed metrics, measurement
evidence, artifacts, lifecycle events, and execution provenance produced by one
run.

## Run record shape

The v1 record contains:

```text
schema
record
  study_name
  run_id
  requested
  resolved
  observed
  status
  metrics
  evidence
  events
  artifacts
  provenance
```

`requested` uses exactly the same `RunSpec` JSON shape as
`metria.study_recipe.v1`. The recipe and run-record serializers therefore share
one RunSpec parser/serializer rather than defining two request schemas.

Metrics retain their complete direct-comparison identity:

```text
name
unit
direction
method
version
```

and can also retain raw samples, aggregation, uncertainty fields, and coverage.
A metric mapping key must match its `MetricDefinition.name`; the parser rejects
ambiguous renamed entries.

## Python API

```python
from metria import (
    dump_run_record,
    load_run_record,
    run_evidence_digest,
    run_record_digest,
)

record = load_run_record("run-0001.json")
print(run_record_digest(record))
print(run_evidence_digest(record))

dump_run_record("normalized-run.json", record)
```

The parser is strict. It rejects unknown schema versions/fields, malformed
metric objects, non-finite numbers, non-string mapping keys, and values that are
not representable by strict JSON.

## Two different digests

`run_record_digest()` hashes the complete canonical versioned record, including
study/run identity and requested intent.

`run_evidence_digest()` hashes produced evidence only:

- resolved state;
- observed state;
- lifecycle status;
- metrics and retained samples;
- measurement evidence;
- events;
- artifact references;
- provenance.

It intentionally excludes `study_name`, `run_id`, and requested intent so
byte-equivalent produced evidence can be correlated independently of local run
naming.

Neither digest is a universal comparability proof. Two records can have related
or identical evidence and still be invalid to compare under a study design. A
`ComparisonPlan` remains authoritative for which dimensions may vary, must
match, or form comparison blocks.

## Compare saved records

Use an explicit study recipe to supply the comparison semantics:

```bash
metria compare run-0001.json run-0002.json --recipe study.json
```

For machine-readable output:

```bash
metria compare run-0001.json run-0002.json \
  --recipe study.json \
  --json
```

More than two run files are allowed. Metria evaluates every pair under the same
recipe `ComparisonPlan`.

The comparison command deliberately requires `--recipe`. It does not invent an
empty/global comparison plan and does not use a digest as a proxy for study
semantics.

Each loaded record must:

1. have the same `study_name` as the recipe; and
2. contain a requested `RunSpec` that appears in the recipe.

This prevents accidentally applying a comparison plan from an unrelated study.

## Exit status

`metria compare` returns:

- `0` when every generated pair is directly compatible under the plan;
- `1` when the records are valid but at least one pair is not directly
  compatible;
- `2` for malformed input, invalid schemas, a record/recipe binding mismatch,
  or another command error.

An incompatible comparison is therefore distinguishable from invalid input in
scripts and CI.

## Metric compatibility

Even when the study-level dimensions match, direct raw metric comparison
requires identical metric identity. For example, two latency values with
methods `wall_clock` and `estimated` are retained as methodologically different
and reported under `incompatible_metrics`.

Cross-method analysis is still possible, but it must be performed by an
explicit named analysis that defines how the methods may be combined. The
saved-record comparison CLI does not silently reinterpret those values.

## Privacy and publication

A run record can contain sensitive material supplied by a runtime, measurement,
or artifact manifest. Metria's built-in runtime/measurement paths try to retain
privacy-conscious evidence, but serialization itself does not redact arbitrary
caller-provided fields.

Review a record before publishing it. A deterministic JSON format makes the
payload reproducible; it does not make every payload public-safe.
