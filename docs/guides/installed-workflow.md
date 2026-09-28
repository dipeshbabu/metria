# From installation to an inference-change decision

These commands are part of the Metria 0.2 development line. Published 0.1.1 keeps
its earlier command surface. Until 0.2 is released, install a built Metria wheel
from this checkout; no component package is needed for the commands below.

## See the result format without a model

```bash
metria demo --case pass --output demo-pass
metria demo --case fail --output demo-fail
metria demo --case not-comparable --output demo-invalid
```

The expected exit codes are 0, 1 and 3. Every bundle and report is explicitly
synthetic. A demo cannot establish runtime support, model quality or performance.
It runs in a separate bounded process so fake evidence never replaces the active
verifier's runtime bindings. Choose a new output directory for each invocation.

## Prepare your real comparison

With a qualified vLLM runtime and pinned local model already installed, run Metria
from that runtime's environment:

```bash
metria recipe prepare-vllm \
  --model /path/to/local/model \
  --descriptor trusted-model-files.json \
  --workload prompts.jsonl \
  --policy policy.json \
  --output study.json

metria recipe validate study.json
metria verify study.json --output verification
```

Preparation verifies model/tokenizer file pins, fingerprints the installed runtime,
and creates reference/candidate configurations for caching off/on. It reports missing
runtime dependencies and malformed inputs before model launch. It never downloads
models or installs inference engines. The recipe contains your workload and any
expected answers; keep it with the input data you control.

The installed wheel includes the tiny SmolLM2 model descriptor and a synthetic
repeated-prefix workload. If those exact model/tokenizer files are already present,
try the contract example without a source checkout:

```bash
metria recipe prepare-vllm --example \
  --model /path/to/pinned/SmolLM2-135M --output example-study.json
metria verify example-study.json --output example-verification
```

Use `--context`, `--max-tokens`, `--warmup-trials`, `--measured-trials`, and `--timeout`
to set explicit limits. Preparation and verification preserve paths containing
spaces or special characters; quote them in your shell. Existing recipes and
evidence directories are never silently overwritten.

See [vLLM profile requirements](vllm-prefix-verification.md) for supported wheel
versions and environment setup, and [task checks and policies](verification-decisions.md)
for exact-answer/structured-output checks and explicit acceptance budgets.

## Read the decision

Open `verification/report.md` and retain `verification.json` with both run records.
Read execution and comparison status before interpreting policy or measurements.
An invalid comparison or missing required evidence cannot become policy PASS.

The report separates token drift, declared task-check outcomes, observed reference
repeatability and compatible latency. A policy pass applies to those user-defined
criteria on this workload. Token differences alone do not establish worse answers,
and tiny-model example timings are not production performance claims.

The [CI guide](verification-ci.md) explains exit codes, summaries and artifact
retention. The [pilot checklist](verifier-pilot.md) records whether the workflow
actually helped a user decide to keep or reject an inference change.
