# Representative maintainer workload

This pilot exercises an installed Metria workflow on public, authored tasks using
the pinned `HuggingFaceTB/SmolLM2-360M-Instruct` model. It records maintainer
validation, not participant feedback or external adoption. The model is small
enough for the existing 4 GiB GTX 1650 and is not a production-quality baseline.

The intended infrastructure change is prefix caching off versus on. Model and
tokenizer contents, runtime, greedy decoding, context and each study's generation
limit remain fixed. Shared system instructions provide a repeated prefix;
retained native cache counts must prove reuse.

The initial preparation attempt rejected a Hugging Face revision-tree cache
receipt as an extra model payload. That failed setup is retained. The inventory
check now recognizes only `.cache/huggingface/trees/<40-hex-revision>.json` as
download metadata. Extra weights, unknown cache JSON and other configuration
files still fail inventory validation; no artifact pins were relaxed.

Acceptance criteria are declared before execution:

- Candidate and reference must pass every configured task check.
- Classification and extraction tasks require exact stripped output.
- Structured-output tasks require a parseable JSON object with declared keys;
  this checks structure, not arbitrary semantic correctness of every value.
- Reports must retain completed native token captures, artifact pins, compatible
  comparisons and observed candidate cache hits.

Two workload groups exercise literal output contracts and representative
classification/extraction/structured-output tasks. A separate constrained run
uses a one-token generation budget with the representative tasks. That budget is
expected to truncate some answers; its failed task checks must remain visible.
It is a negative setup test, not evidence of model degradation caused by caching.

Each completed comparison gets a `metria pilot record` receipt with elapsed setup
time, retained report hashes, predeclared criteria, observed friction and a
keep/reject/inconclusive decision. `would_reuse` remains null because an automated
maintainer run cannot provide a participant's preference. Successful comparisons
only establish those workload contracts on this device. Failed tasks must be
fixed in the model, prompt or generation setup before considering a production
deployment; output token agreement alone is insufficient.

Actual participant feedback is still outstanding in
[issue #159](https://github.com/dipeshbabu/metria/issues/159). Use the
[pilot protocol](verifier-pilot.md) to collect it with permission.
