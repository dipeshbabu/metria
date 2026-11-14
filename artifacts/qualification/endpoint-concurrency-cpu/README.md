# Native CPU batch invariance on vLLM endpoints

On 2026-11-13 `remote_endpoint_change.v1` with a declared concurrent schedule
(`--concurrency 4 --concurrency-rounds 3`) ran against two `vllm serve`
processes from the pinned `0.30.0+cpu` wheel in WSL2. Both served the pinned
SmolLM2-135M in float32 with `--max-num-seqs 4`, so the engine could batch up to
four requests, on separate CPU pairs. The workload is the same four prompts as
[`endpoint-cpu`](../endpoint-cpu/README.md): 16 greedy tokens, seed 42, two
sequential measured trials, then 12 concurrent requests per run.

| Run | Concurrent requests matching sequential output | Max requests in flight (client) | Requests/s | Mean request s |
| --- | --- | --- | --- | --- |
| reference | 12 / 12 | 4 | 1.275 | 3.099 |
| candidate | 12 / 12 | 4 | 2.637 | 1.452 |

Verdict **PASS**: batch invariance was 1.0 for both runs, so this model,
engine build and CPU backend produced identical greedy tokens with and without
concurrent batching.

**The client throughput difference is not a property of the change.** Both
servers had identical configurations; the 2× gap comes from CPU contention and
warm-up on a shared WSL host. `client.*` ratios include everything between the
client and the server, so they are meaningful only when both deployments are
measured from the same place under comparable load.

Files: the unmodified verification bundle (`verification/`), the prepared
`study.json`, `policy.json` and `workload.jsonl`, and logs with local paths
replaced by placeholders.
