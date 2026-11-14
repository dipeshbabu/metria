# Native CPU qualification of remote endpoint verification (staged)

On 2026-11-13 `remote_endpoint_change.v1` ran end to end against real
`vllm serve` processes from the pinned vLLM `0.30.0+cpu` wheel in WSL2 (x86_64),
using `metria recipe prepare-endpoint` and `metria verify` exactly as
documented. The model is the pinned SmolLM2-135M (revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`), served as `smollm2` with eager
execution, one sequence, and two OpenAI-compatible servers bound to separate
CPU pairs. The workload is four prompts with a `nonempty` task check, 16 greedy
tokens (seed 42), no warmup and two measured trials, with staged top-20
logprobs for trajectory KL divergence.
[`qualify_endpoint_cpu.sh`](../../../tools/qualification/staged/qualify_endpoint_cpu.sh)
reproduces every case.

| Case | Servers | Declared | Verdict | Why |
| --- | --- | --- | --- | --- |
| `identical` | float32 vs float32 | `dtype=float32` on both | **PASS** | Agreement 1.0, trajectory KLD 0, repeatability 1.0 |
| `bf16` | float32 vs bfloat16 | `dtype` float32 → bfloat16 | **PASS** | Agreement 0.953, KLD 0.0559 nats, within the declared policy (≥ 0.9, ≤ 0.1) |
| `contradiction` | float32 vs float32 | candidate `engine_version=0.29.0` | **EXECUTION_FAILED** | The server reported `0.30.0`; launch was refused before any prompt |
| `undeclared` | `max_model_len` 256 vs 512 | nothing differs | **NOT_COMPARABLE** | Only the undeclared server-reported `max_model_len` differed |

**Cross-check with the in-process engine.**
[`crosscheck-in-process.json`](crosscheck-in-process.json) compares the
endpoint reference run's token IDs with the in-process vLLM run retained in
[`vllm-staged-treatments-cpu`](../vllm-staged-treatments-cpu/README.md) (same
model, prompts and greedy settings): all four prompts match. The `bf16` case
also reproduces the in-process float32 → bfloat16 result in
[`vllm-trajectory-kld-cpu`](../vllm-trajectory-kld-cpu/README.md) exactly
(KLD 0.05592205681797503 nats, agreement 95.3125), so the HTTP path adds no
tokenization or decoding difference here.

**Identity as observed.** Every run records `identity_authority: claimed`.
`/version`, `/v1/models` (id, `max_model_len`, root kept only as a hash) and
`/tokenize` were exposed by the server; the tokenizer agreed on every prompt and
identity snapshots were stable in all completed runs.

**Not covered natively.** Identity drift during a run is covered by unit tests
only: restarting a server mid-run makes requests fail before the second
snapshot. `kv_cache_dtype` and `hardware` cannot be read from a vLLM server, so a
false declaration of those is not detectable; the report says identity is
claimed for this reason. The GPU case (FP8 KV cache with a KLD policy on an L4)
is still pending.

## Files

- `<case>/`: the unmodified `metria verify` bundles (`verification.json`,
  `report.md`, both run records). Local paths are left as recorded so record
  and evidence digests stay valid.
- `<case>.study.json`, `policy-*.json`, `workload.jsonl`: the prepared inputs.
- `<case>.log`, `serve-*.log`: preparation, verification and server logs, with
  local paths replaced by `${HOME}`, `${RUNTIME_INSTALL}`, `${HF_CACHE}` and
  `${OUT}`.
- `files-sha256.json`: SHA-256 of every file above.
