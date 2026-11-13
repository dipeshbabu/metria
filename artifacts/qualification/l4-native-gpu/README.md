# Native NVIDIA L4 observations: FP8 KV cache and staged vLLM treatments

On 2026-11-13 one rented NVIDIA L4 (Ada, compute capability 8.9, 22.5 GB, driver
595.84, CUDA runtime 12.9) ran Metria source
`363ef4a017146808a5128d8343d8ffd45404a770` with the pinned vLLM
`0.30.0+cu129` wheel (SHA-256 `e98cb696…aabcc`) and the pinned SmolLM2-135M
files at revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. Every run used
Metria's real vLLM adapter and study lifecycle: float16, eager execution, one
sequence, a 128-token model limit, 65% GPU-memory utilization, greedy decoding,
context 96 and 16 generated tokens. Model files remain outside the repository.

These are retained native observations. They do not promote a `metria verify`
profile, and no acceptance policy was evaluated.

## Adapter qualification on SM89

[`adapter/`](adapter/qualification.json) is the standard
`tools/qualification/qualify_vllm.py --backend cuda` run on this device: two plain
greedy prompts, an actual reset, repeated capture, observation and cleanup. It
completed. It extends the retained [GTX 1650 run](../vllm-0.30.0/README.md) to an
SM89 device under the same pinned wheel.

## FP8 KV cache applies and changes behavior

[`fp8-kv-cache/`](fp8-kv-cache/comparison.json) compares a reference run
(`vllm.kv_cache` dtype `auto`) with a candidate (`fp8`) over four prompts.

- Requested → resolved → observed: the candidate requested `fp8`, resolved
  `kv_cache_dtype=fp8`, and the constructed engine reported
  `cache.cache_dtype = fp8`. The reference reported `auto`. This is the positive
  native FP8 cache observation that the [SM75 failure](../fp8-unsupported/preflight.json)
  could not provide.
- Behavioral change: trajectory agreement score **48.4 / 100**. `capital` and
  `arithmetic` diverge at the first generated token, `story` at token 13, and
  `dog` matches all 16 tokens.

The divergence is a measured property of this tiny model with an uncalibrated
FP8 KV cache. It is not a general FP8 quality claim for other models, scaling
methods or attention backends.

## Staged vLLM treatments apply as requested

[`staged-treatments/`](staged-treatments/staged.run.json) runs the staged
`vllm.scheduler` (`enable_chunked_prefill=true`, `max_num_batched_tokens=2048`)
and `vllm.cache_layout` (`block_size=32`) treatments with the
`vllm.staged_treatments` capability override. All three settings were read back
from the engine as `applied` (`scheduler_config.*`, `cache_config.block_size`).
Greedy trajectories match the reference exactly (agreement 100), as expected
for scheduling and memory-layout changes. CPU-lane behavior remains unobserved.

## Environment and deviations

[`setup.log`](setup.log) records preparation; local paths are replaced with
`${WORK}`. The wheel hash was verified before installation and every model file
was verified against its descriptor hash before each run.

Compared with the retained GTX 1650 environment, the resolver selected newer
`transformers` 5.22.2 (was 5.17.0), `huggingface_hub` 2.4.0 (was 1.33.0) and
`tokenizers` 0.23.3 (was 0.23.2). `torch` 2.13.0+cu129, `triton` 3.7.1,
`flashinfer-python` 0.6.18.post1 and the `setuptools` 84.0.0 override match.
Full package lists are in each record.

[`files-sha256.json`](files-sha256.json) lists the retained file bytes.

## Not established

- A qualified FP8 `metria verify` profile: that still needs an FP8 route with
  task checks and an explicit acceptance policy.
- Any FP8 quality or performance claim beyond these four prompts.
- Staged-treatment behavior on CPU, on other GPUs, or for other values.
- Trajectory KL divergence on vLLM; the staged top-k capture exists only for
  llama.cpp.
