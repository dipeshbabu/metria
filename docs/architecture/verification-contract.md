# The public verification contract

The first end-to-end Metria command is:

```bash
metria verify study.json --output verification --json
```

It consumes a `metria.study_recipe.v1` recipe containing exactly two ordered
runs: reference, then candidate. The qualified public scope is a local llama.cpp
CPU thread change with one pinned model and capture provider, plain greedy
completion, and one trajectory measurement/analysis route. Additional qualified
Python adapter evidence does not automatically widen the CLI scope.

The preparation tool emits exactly these intended-change paths:

```text
runtime.threads
resolved.runtime.threads
observed.runtime.threads
observed.identity.applied.fields.threads
```

The local scope rejects broad parent variations, incomplete/unknown variation
paths, and comparison waivers before execution. Allowing an entire observed
identity subtree to vary could otherwise hide a difference unrelated to threads.
General `ComparisonPlan` semantics, including explicit waivers, remain available
through the supporting Python/saved-record APIs; they do not redefine this
qualified CLI contract.

## One lifecycle and one result

The command reuses `StudyRecipe`, `StudySpec`, `RunSpec`, `execute_study()`, runtime
adapters, measurement requirements, pairwise analysis, and `RunRecord` persistence.
It does not introduce a second orchestration graph or auto-load third-party plugins.
Routes/configuration are validated before launch; capture requirements are
negotiated during preflight. Reference evidence is persisted before candidate
execution. Failed, partial, timed-out, and interrupted records remain inspectable.

Behavioral analysis requires completed runs, sufficient identity/capture evidence,
and a valid comparison. Method-compatible performance deltas and explicit user
policy follow those gates. Neither policy nor a broad variation declaration can
turn missing required evidence into success.

`verification.json` is the versioned canonical result, published last;
`manifest.json` is a compatibility alias. It identifies the recipe, implementation,
hardware evidence, run-record/evidence digests, comparison, analysis, and policy.
`report.md` is a deterministic projection. Output directories must be new;
completed records survive later persistence failures without a success result.

The stable exit mapping and redacted machine-readable input errors are described
in [CI integration](../guides/verification-ci.md). The
[local guide](../guides/metria-verify.md) documents preparation and interpretation.
Fixture/contract tests cover routes, capture negotiation, all verdict states,
privacy boundaries, deterministic persistence, and output collisions. Pinned
[real-engine evidence](../guides/runtime-qualification.md) remains a separate level
of confidence.
