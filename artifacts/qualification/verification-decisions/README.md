# Native decision-policy qualification

These installed-wheel CPU runs used source
`a3e61f38e0251ad25f434b6459ca9e4183fff8b8`, the pinned vLLM 0.30.0+cpu environment,
and the same SmolLM2-135M model/tokenizer files as the
[prefix-cache qualification](../vllm-prefix-cache/README.md).

Both runs evaluated a nonempty-response check and a deliberately incorrect
exact-text expectation on two synthetic prompts across two measured trials.
The checks are functional tests of the evidence/policy path, not a model-quality
benchmark. Expected and generated text do not appear in the summary evidence.

| Explicit policy | Expected and observed result |
|---|---|
| Candidate task-check pass rate at least 0.5 | [PASS, exit 0](pass/report.md) |
| Candidate task-check pass rate at least 1.0 | [FAIL, exit 1](fail/report.md) |

Both reference and candidate completed with valid comparison/evidence. Reports
separate token drift, declared task checks, observed reference repeatability and
compatible latency. The deliberately wrong check remains visible even in the
permissive policy case; PASS means that stated threshold was met.

Original recipes, reports and run/result bytes are retained with shared provenance
manifests. `qualification.json` identifies the installed wheel and source. The
environment retains the documented setuptools84 override and CPU OpenMP preload.
No runtime/model binaries are redistributed. No additional license is asserted for
model-derived output token IDs. These results do not establish deployment safety,
statistical significance, external-user adoption or general model quality.
