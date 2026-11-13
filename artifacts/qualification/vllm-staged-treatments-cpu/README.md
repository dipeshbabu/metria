# Native vLLM CPU observations for staged treatments

On 2026-11-13 the pinned vLLM `0.30.0+cpu` wheel (SHA-256 `0ee75278…ed78`) ran
through Metria's real vLLM adapter and study lifecycle on Linux (WSL2, x86_64)
with PyTorch `2.13.0+cpu`, `intel-openmp` 2024.2.1 (`libiomp5` preloaded),
two OpenMP threads bound to CPUs 0–1, 1 GiB of KV cache, float32, eager
execution, one sequence and a 128-token model limit. The model is the pinned
SmolLM2-135M at revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`; every file
was verified against its descriptor hash before each run. Local paths are
replaced with `${MODEL_SNAPSHOT}`, `${MODEL_CACHE}`, `${HUB_CACHE}`,
`${RUNTIME_ENV}` and `${RUNTIME_INSTALL}`.

| Run | Treatments | Outcome |
| --- | --- | --- |
| `reference` | `vllm.kv_cache` `auto` | completed |
| `staged` | `vllm.scheduler` (`enable_chunked_prefill=true`, `max_num_batched_tokens=2048`), `vllm.cache_layout` (`block_size=32`) | completed; all three settings read back from the engine as `applied`; greedy trajectories match the reference (agreement 100) |
| `staged-reject` | `vllm.scheduler` (`enable_chunked_prefill=false`, `max_num_batched_tokens=64`) | **rejected by the engine**: vLLM's configuration validation raised `ValidationError` at launch, after warning that disabling chunked prefill is not officially supported for this model; the record ends `failed` at the `launch` stage with no measurement |

The rejection case asks for a 64-token batch budget below the 128-token model
length without chunked prefill, which cannot schedule a full prompt. It shows the
negative path: an invalid staged value fails before any measurement instead of
running with a silently different configuration. Error text is retained only as
a hash in the record, as for every Metria lifecycle error.

Together with the [L4 CUDA observations](../l4-native-gpu/README.md), the staged
`vllm.scheduler` and `vllm.cache_layout` treatments are observed applied on
both pinned lanes. Compared with the retained CPU qualification, the resolver
selected newer `transformers` 5.22.2 (was 5.17.0) and `triton` 3.9.0
(was 3.8.0); torch, intel-openmp and the setuptools 84.0.0 override match.
Full package lists are in each `*.observation.json`.

[`files-sha256.json`](files-sha256.json) lists the retained file bytes.

## Not established

The treatments remain staged: no verify profile accepts them and no policy
target consumes them. These runs do not cover other models, values, CPU
topologies or performance effects.
