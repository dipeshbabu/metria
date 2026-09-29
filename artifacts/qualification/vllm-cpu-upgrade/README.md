# Installed-wheel vLLM CPU upgrade qualification

On 2026-09-28, the same installed Metria wheel prepared and executed a reference
in vLLM `0.29.0+cpu` and a candidate in `0.30.0+cpu`. Both native runs completed,
the comparison was valid, and the predeclared completion-presence policy returned
`PASS`. Both workers confirmed the same controlled CPU set, model/tokenizer
contents, generation settings and cache-reset trial method.

- Metria source: `23f25f2b3f930c49ff7f8d5aeefa822be5c48d8a`.
- Installed wheel SHA256: `3d20f24e5f7f18ac43db045b562b621227b7adbb9caf3309973c20037c849cd3`.
- Model: pinned SmolLM2-135M files at revision
  `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
- Two separate runtime prefixes, same Python and Metria contents, float32,
  one worker, eager greedy completion, context 256, at most eight generated tokens.
- One warmup and two measured trials over two public story prefixes. Prefix
  caching is disabled in both environments; native reset acknowledgements and
  token/cache counters are retained.

[qualification.json](qualification.json) records the source/wheel identity and
limitations. The actual [recipe](study.json), [workload](workload.jsonl), and
[policy](policy.json) are preserved. Both environment descriptors and native
placement observations appear in the [reference](verification/reference.run.json)
and [candidate](verification/candidate.run.json) records. The
[report](verification/report.md), [canonical result](verification/verification.json),
and [file hashes](files-sha256.json) retain the completed decision and original
file bytes.

The controller restricts each interpreter before engine startup; a named installed
worker extension independently reads all native worker thread affinities before
and after inference. The recipe's environment descriptors describe observed
availability before this deliberate binding. No callable/pickle RPC fallback is
enabled. Every returned record is bound to its original request and environment.

This compares the pinned installed runtime stacks, including their recorded
dependency and OpenMP-library differences. It does not isolate vLLM source alone.
The small workload qualifies the execution/evidence interface, not arbitrary
models, production performance or external adoption. Presence checks do not
establish answer correctness, and loaded-engine request latency is not serving
throughput or streaming TTFT. Model files remain outside the repository.
