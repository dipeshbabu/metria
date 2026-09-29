# Decision: one inference change verifier

Status: accepted. This records the product decision in
[issue #50](https://github.com/dipeshbabu/metria/issues/50).

Metria verifies changes to LLM inference systems. Its primary workflow takes
one reference and one candidate, retains evidence of what ran, rejects an
unjustified comparison, and reports the measured change under the user's policy.
It sits above inference engines and uses their native observations where available.

## User result and implementation boundary

The result answers four questions: **CHANGE** describes the intended variation;
**EVIDENCE** explains what ran and what remained controlled; **IMPACT** reports
named behavioral and compatible systems measurements; **VERDICT** separates
execution, comparison validity, and policy acceptance.

The existing architecture remains authoritative:

```text
Study = factors + controls + ComparisonPlan
Run = system under test × workload × measurement @ observed environment
RunRecord = requested → resolved → observed → evidence

reference + candidate + declared change
  → existing execute_study / execute_run lifecycle
  → evidence sufficiency and fail-closed comparison
  → named/versioned analyses and compatible systems measurements
  → optional user policy
  → verification.json + report.md + both RunRecords
```

`StudyRecipe`, `StudySpec`, `RunSpec`, runtime/session protocols, measurements,
pairwise analyses, and versioned persistence serve this workflow. General Python
composition remains useful; it does not require a second public orchestration
model. `recipe`, `inspect`, and saved-record `compare` are supporting operations.

## Current delivery and scope

The [local verifier](../guides/metria-verify.md) qualifies a pinned llama.cpp CPU
thread change with the same immutable GGUF model and workload. It records native
decode-time token trajectories, applied thread/context evidence, behavioral
divergence, explicit policy decisions, and cold-process request latency when
compatible native timing exists. [CI integration](../guides/verification-ci.md)
retains the same result and stable exit codes.

This delivery has explicit limits:

- Metria 0.1.2 also qualifies a [local vLLM prefix-cache comparison](../guides/vllm-prefix-verification.md),
  backed by retained installed-wheel CPU/CUDA evidence. That scope does not
  qualify runtime upgrades, FP8 precision, or arbitrary vLLM recipes.
- Independently qualified [llama.cpp CPU builds](../guides/llamacpp-build-verification.md)
  can be compared with the same model, generation, workload and thread controls.
- A [GGUF Q8_0 weight conversion](../guides/gguf-quantization-verification.md)
  can be verified on CPU with unchanged tokenizer, model layout, runtime and workload.
  Reports retain the actual mixed tensor-storage inventory.
- A [pinned vLLM CPU runtime-stack upgrade](../guides/runtime-upgrade-verification.md)
  uses separate interpreters, independently checked environment identities and
  native CPU placement with fixed model/tokenizer and workload controls.
- FP8 KV precision changes remain
  [staged templates](../../examples/verification/README.md) until the full intended
  change, observed identity, workload, and comparison path are qualified.
- A separate [local serving profile](../guides/serving-measurements.md) measures
  streamed token delivery, bounded concurrency, throughput, and native CUDA
  allocator/KV storage. It excludes HTTP/network overhead; individual-token
  intervals remain unavailable when the runtime coalesces output chunks.
  Cold-process latency retains its distinct startup/model-loading boundary.
- `VERIFIED` means valid completed comparison and analysis. A policy is required
  for `PASS`/`FAIL`; no universal quality, safety, or performance threshold is implied.

Closing the product design decision establishes these boundaries. It does not
declare every roadmap use case supported or every runtime identity authoritative.

## Evidence requirements for extending support

Each added verification path must retain what was requested, what immutable
artifacts and configuration were resolved, and what the runtime independently
observed. It must prove the intended change applied and identify all controls,
missing facts, contradictions, and undeclared differences before analysis.

The result identifies measurement method/version, behavioral divergence,
compatible systems impact, and each evaluated acceptance criterion. Missing
evidence remains unknown or unavailable. A digest identifies serialized content;
it cannot establish semantic comparability. A policy cannot make an invalid
comparison valid. Reports preserve privacy-conscious hashes and avoid raw prompts
or credentials by default.

Support expands through the same adapter and evidence contracts, with negative
semantic tests, retained pinned engine/hardware runs, and a copyable example.
No adapter is added merely to increase the runtime count.

## Ownership and project scope

| Area | Responsibility |
|---|---|
| Metria core | Verification, shared lifecycle, evidence, identity, capability/preflight, comparison, records, and result semantics. |
| KV Fidelity | Built-in Metria fidelity methods with preserved method identities; the verifier exposes trajectory semantics and expert commands use `metria fidelity`. |
| TurboQuant Reference | Independently versioned source-only research implementation, pinned by immutable repository revision. |
| Explicit integrations | Optimization-specific support knowledge; generic core retains common evidence and capability semantics. |
| Research and artifacts | Dated conclusions and retained evidence with provenance and limitations. |
| Tools | Thin or specialized utilities using the owning libraries; no parallel runtime/evaluator architecture. |

Broad sweeps, distributed scheduling, automatic search/recommendations, Pareto
optimization, generic benchmark products, arbitrary plugin auto-discovery,
universal fidelity scores, broad training optimization, all-runtime environments,
and hosted dashboards remain deferred product directions.

Issue triage and PR review ask whether a change makes one inference change easier
to verify, makes its evidence more trustworthy, or improves the keep/reject
decision. Work outside those questions needs a focused component/research purpose
or a new product decision. Package publication remains a separate protected
release procedure; this decision makes no package-index availability promise.
