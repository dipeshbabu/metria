# Verify a change between two running servers (staged)

`remote_endpoint_change.v1` compares a **reference** and a **candidate**
inference server through their OpenAI-compatible completions APIs. Use it for
changes you deploy as a server: a new engine version, a different KV-cache
dtype, a quantized checkpoint, or new server flags.

!!! warning "Staged"
    This profile is staged until the native evidence in
    [#214](https://github.com/dipeshbabu/metria/issues/214) is retained. Server
    identity is **claimed**: Metria records what each server reports about
    itself and checks it stays consistent, but cannot verify which weights or
    binary a remote server runs.

## Requirements

- Two running servers with a flavor that returns token IDs (`vllm`), reachable
  over `https`, or `http` on loopback.
- A local copy of the tokenizer the servers use. Its files are hashed into the
  recipe and used to tokenize every prompt.
- A JSONL workload whose prompts declare task checks, and a policy JSON file.
  The result is always `PASS` or `FAIL`.

## Prepare and verify

```bash
metria recipe prepare-endpoint \
  --model-id served-model-name \
  --tokenizer /path/to/tokenizer-dir \
  --workload tasks.jsonl \
  --policy policy.json \
  --reference-url-env METRIA_REFERENCE_URL \
  --candidate-url-env METRIA_CANDIDATE_URL \
  --reference-key-env METRIA_REFERENCE_KEY \
  --candidate-key-env METRIA_CANDIDATE_KEY \
  --reference-declared kv_cache_dtype=auto \
  --candidate-declared kv_cache_dtype=fp8 \
  --output endpoint-study.json

export METRIA_REFERENCE_URL=https://reference.example.internal
export METRIA_CANDIDATE_URL=https://candidate.example.internal
metria verify endpoint-study.json --output endpoint-verification
```

The recipe stores only environment-variable names. URLs and keys never enter
the recipe, run records, reports or error messages.

## Declaring the change

`--reference-declared` and `--candidate-declared` (repeatable `KEY=VALUE`)
state what each deployment is. Allowed keys: `engine`, `engine_version`,
`model_root`, `model_revision`, `max_model_len`, `dtype`, `kv_cache_dtype`,
`tensor_parallel_size`, `pipeline_parallel_size`, `hardware`.

- Keys whose values differ are the intended change. Preparation derives
  `comparison.vary` from them; the verifier rejects any other `vary`.
- `engine_version`, `max_model_len` and `model_root` are also checked against
  what the server reports: a contradiction fails the run before any prompt.
  When one of these is the declared change, the matching server-reported
  identity is allowed to differ.
- Any **other** difference the servers report, such as an engine version you
  did not declare, makes the runs `NOT_COMPARABLE`.

## What the verifier checks

| Check | Outcome when it fails |
| --- | --- |
| Same tokenizer pins, workload, scenario (temperature 0 and a seed) and runtime settings, apart from endpoint variables and declarations | Rejected before execution |
| Server identity unchanged between launch and the last request | `INSUFFICIENT_EVIDENCE` |
| Server `/tokenize` agrees with the pinned tokenizer | `INSUFFICIENT_EVIDENCE` |
| Echoed prompt token IDs match the request | The run fails |
| Every request returned complete token IDs (and top-20 logprobs with `--staged-kld`) | The run fails |
| Task checks retained for every measured trial | `INSUFFICIENT_EVIDENCE` |

Trials run sequentially. Server state, including any prefix cache, is not
reset between trials, and the workload evidence says so.

## Concurrency and batch invariance

Production servers batch concurrent requests, and batching can change the
numerics of a forward pass. Add `--concurrency N` (2–64) and optionally
`--concurrency-rounds R` (1–10, default 2) to `prepare-endpoint`. After the
sequential trials, each run sends every prompt `R` times with up to `N`
requests in flight, and compares each output with the same prompt's first
sequential output. The schedule is part of both runs' trial policy.

| Policy target | Meaning |
| --- | --- |
| `behavior.reference_batch_invariance`, `behavior.candidate_batch_invariance` | Fraction of concurrent requests whose token IDs equal the sequential output |
| `client.requests_per_second_ratio` | Candidate/reference throughput of the concurrent pass, higher is better |
| `client.request_latency_ratio` | Candidate/reference mean request time in the concurrent pass, lower is better |

All four use analysis `metria.batch_invariance` version `1`. The `client.*`
targets are measured on the client clock and **include the network and
anything else sharing the server**, so use them only between deployments
measured from the same place under comparable load. A server started with a
single sequence slot (vLLM `--max-num-seqs 1`) queues requests instead of
batching them, which makes invariance trivially 1.

On real CPU servers with `--max-num-seqs 4`, the
[retained run](../../artifacts/qualification/endpoint-concurrency-cpu/README.md)
measured batch invariance 1.0 for both deployments (12 of 12 concurrent
requests), while client throughput differed 2× between two identically
configured servers because of host contention.

## Decision

Trajectory agreement, task-check pass rates and reference repeatability (with
two or more measured trials) feed the [policy](verification-policies.md).
`--staged-kld` adds the staged `behavior.trajectory_kld_*` targets. Latency is
reported as **client-side** request time, including the network, load balancer
and any other traffic on the server; it is descriptive and does not feed
`performance.*` targets.

## Retained evidence

[`artifacts/qualification/endpoint-cpu/`](../../artifacts/qualification/endpoint-cpu/README.md)
runs this profile against real `vllm serve` processes (pinned `0.30.0+cpu`,
SmolLM2-135M):

- identical servers PASS with agreement 1.0 and trajectory KLD 0;
- float32 → bfloat16 PASS a declared policy with agreement 0.953 and KLD
  0.0559 nats, exactly matching the in-process vLLM result;
- a declared engine version the server contradicts ends `EXECUTION_FAILED`;
- an undeclared `max_model_len` difference ends `NOT_COMPARABLE`.

Endpoint token IDs match the in-process engine for every prompt. The GPU case
is pending.

See the [endpoint adapter](endpoint-runtime.md) for request, retry and privacy
details.
