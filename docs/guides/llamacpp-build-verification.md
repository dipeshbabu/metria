# Verify a llama.cpp build change

Metria's 0.2 development CLI compares two independently qualified CPU capture
providers with the same immutable GGUF, workload, generation settings and thread
counts. Only the provider directory and its binary identities may vary. This
scope does not qualify GPU offload, different model files or arbitrary runtime
settings.

Build both providers with the [native capture patch](../../tools/qualification/README.md).
Keep separate binary directories and record each source revision, patch digest,
compiler and build flags. Two paths containing identical provider bytes are not
a build change.

Prepare a JSONL workload with `id`, `prompt` and optional
[task checks](verification-decisions.md). From the installed Metria environment:

```bash
metria recipe prepare-llamacpp-build \
  --reference-bin-dir /path/to/reference/bin \
  --candidate-bin-dir /path/to/candidate/bin \
  --model /path/to/pinned/model.gguf \
  --model-sha256 TRUSTED_MODEL_SHA256 \
  --workload prompts.jsonl \
  --policy acceptance-policy.json \
  --output build-study.json

metria verify build-study.json --output build-verification
```

Preparation hashes each provider, performs a native capture probe for each, and
saves `build-study.reference.qualification.run.json` and
`build-study.candidate.qualification.run.json`. A failed probe is retained and
prevents publication of the recipe. Use fresh output paths. The workload and
expected answers are stored in the recipe; keep it with the input data you control.

The verifier checks each provider pin before resolution/launch and again when
collecting observations. Both runs must confirm the requested thread counts,
context and plain-completion mode and provide nonempty native sampled token IDs.
Unexpected model, tokenizer/vocabulary, configuration or capture differences
cannot become a valid comparison simply because the binary changed.

The report retains both run records and distinguishes comparison validity from
user acceptance. Without a policy, a completed valid pair is `VERIFIED`. With
one, Metria evaluates the declared criteria and reports `PASS` or `FAIL`.
Compatible cold-process request latency includes process startup and model
loading. It is not TTFT, loaded-engine serving throughput or a general speedup
claim. Each native request has a deadline (`--timeout` during preparation).

[Retained installed-wheel qualification](../../artifacts/qualification/llamacpp-builds/README.md)
compares a retained O3 provider against an independently built O2 provider from
the pinned llama.cpp source and capture patch. Both native qualification probes
and the complete verification passed. The tiny workload establishes the command
and evidence contract; it is not an external pilot or a production benchmark.
