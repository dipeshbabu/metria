# Streaming, concurrency and memory measurements

Use `metria recipe prepare-vllm-serving` to compare one versus two in-flight
requests through the pinned vLLM `0.30.0` native `AsyncLLM.generate` interface.
Both runs keep the engine capacity at two, disable prefix caching, use the same
immutable model/tokenizer files and runtime installation, and perform greedy
single-device inference. This measures the local serving engine API. It does
not measure an HTTP endpoint or network latency.

From the qualified Linux/WSL vLLM environment:

```bash
metria recipe prepare-vllm-serving \
  --model /path/to/model --descriptor model-manifest.json \
  --workload workload.jsonl --policy policy.json \
  --warmup-trials 1 --measured-trials 3 --output serving-study.json
metria verify serving-study.json --output serving-evidence
```

Provide at least two JSONL prompt rows. Each row accepts the same explicit
`checks` as the [task-quality workflow](verification-decisions.md). The model
manifest contains trusted file-to-SHA256 pins. Preparation verifies these pins
and the installed runtime; base Metria does not install an inference engine.

The report keeps these measurements separate:

| Measurement | Boundary and aggregation |
| --- | --- |
| Request latency | Admission into `AsyncLLM.generate` to stream completion; mean over measured requests |
| TTFT | Admission to the first observed native output token chunk; mean over measured requests |
| Inter-token latency | Mean time between individual native token deliveries; unavailable if any request has coalesced chunks or fewer than two tokens |
| Decode rate | Tokens after the first divided by summed per-request token-delivery intervals; unavailable for coalesced chunks |
| Output/request throughput | Completed native tokens/requests divided by the sum of whole measured batch windows |
| Peak allocated/reserved bytes | Maximum native worker PyTorch CUDA allocator peak across measured trials |
| Allocated KV bytes | Maximum total unique native KV tensor storage across measured trials |

All timings use `perf_counter_ns` in the client process. Request latency and TTFT
exclude waiting for the client concurrency slot. Whole batch throughput includes
that waiting. Engine startup and resets are outside the measurement window.
Warmup trials are excluded; the public prefix-cache reset runs before each trial,
and native allocator peaks reset before each measured trial.

Memory observations cover the single inference worker's PyTorch allocator. They
exclude other processes and allocations outside PyTorch. KV allocation measures
capacity, not live occupied tokens. CPU memory and unavailable native allocation
layouts remain unavailable. None of these observations implies production
capacity or a statistically established speedup.

Policy targets use the `serving.` prefix, metric name, and either `_candidate`
(absolute candidate value in the documented unit) or `_ratio`
(candidate/reference). For example, `serving.ttft_seconds_candidate`,
`serving.output_tokens_per_second_ratio`, and
`serving.allocated_kv_bytes_candidate` have target version `1`. A ratio is
unavailable for a zero baseline. Missing or incompatible measurements cannot
satisfy a policy. Keep task-quality criteria separate, such as
`quality.candidate_pass_rate`.

Each bundle retains native token trajectories, finished-request observations,
stream chunk counts and timing offsets, observed concurrency, reset evidence,
native memory receipts, method/version identities and workload hashes. Comparison
requires matching methods, units, complete coverage and fixed controls.
