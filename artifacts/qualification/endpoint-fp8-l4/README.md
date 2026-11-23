# Native GPU remote endpoint verification: automatic vs FP8 KV cache

On 2026-11-23 `remote_endpoint_change.v1` ran against two `vllm serve` processes
from the pinned `vllm==0.30.0+cu129` wheel on a host with two NVIDIA L4 GPUs
(compute capability 8.9): the reference on GPU 0 with the automatic KV cache and
the candidate on GPU 1 with `--kv-cache-dtype fp8`. Both served the pinned
SmolLM2-135M (revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`) in float16
as `smollm2`, with eager execution and one sequence slot.

`metria recipe prepare-endpoint` declared `kv_cache_dtype` `auto` → `fp8` and
`engine_version` `0.30.0` on both runs, with staged top-20 distributions
(`--staged-kld`). The workload is four prompts with a `nonempty` check, 16 greedy
tokens and two measured trials. [`qualify_endpoint_fp8_gpu.sh`](../../../tools/qualification/staged/qualify_endpoint_fp8_gpu.sh)
reproduces it.

| Check | Result |
| --- | --- |
| Verdict | **PASS** under [`policy.json`](policy.json): candidate pass rate ≥ 1.0, trajectory KLD ≤ 0.5 nats |
| Comparison | `VALID`; only the declared `kv_cache_dtype` and endpoint plumbing differed |
| Server identity | claimed `0.30.0` on both, matching the declaration; snapshots stable |
| Candidate pass rate | 1.0 |
| Trajectory KL divergence | 0.105 nats (mean) |
| Trajectory agreement | 48.4 / 100 |
| Candidate server log | `Using fp8 data type to store kv cache` |

FP8 KV caches move SmolLM2-135M's distributions far enough that greedy decoding
diverges early on most prompts (agreement 48.4, consistent with the earlier
in-process [L4 observations](../l4-native-gpu/README.md)). The task checks still
pass, and the mean KLD of 0.105 nats stays under this policy's 0.5 bound.
Whether that is acceptable is a policy choice; the evidence makes the size of
the change explicit. An earlier identical run on a different 2×L4 host produced
the same KLD to every printed digit.

Server identity remains claimed, not verified: vLLM does not report its KV-cache
dtype, so the `fp8` declaration is confirmed by the server's own startup log
rather than by the protocol.

Files: the unmodified verification bundle (`fp8/`), `fp8.study.json`,
`workload.jsonl`, `policy.json`, and logs with the working directory replaced by
`${WORK}` (original hashes in [`originals-sha256.json`](originals-sha256.json));
[`files-sha256.json`](files-sha256.json) hashes every file.
