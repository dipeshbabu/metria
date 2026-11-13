# FP8 KV-cache verification on NVIDIA L4

On 2026-11-13 an NVIDIA L4 (Ada, compute capability 8.9, driver 595.84, CUDA
runtime 12.9) ran the complete `local_vllm_kv_cache_precision.v1` profile from
Metria source `50da19e84586d68845b5df845b24934472da09a1`, using the pinned
`vllm==0.30.0+cu129` wheel and the bounded native worker.

Inputs:

- Model: `HuggingFaceTB/SmolLM2-360M-Instruct` at revision
  `a10cc1512eabd3dde888204e902eca88bddb4951`, verified against the
  [pilot manifest](../maintainer-pilot/representative-tasks/model-manifest.json).
- Workload: the public [representative pilot tasks](../../../examples/pilot/representative-tasks.jsonl)
  (classification, extraction and structured output with task checks), one
  warmup and two measured trials, 32 generated tokens, context 512.
- Policy: the [pilot policy](../../../examples/pilot/policy.json), which requires
  every reference and candidate task check to pass.
- Recipe: [`study.json`](study.json), produced by
  `metria recipe prepare-vllm --kv-cache-dtype fp8`.

## Result: FAIL with a valid comparison

The [report](verification/report.md) and [canonical result](verification/verification.json)
record:

- **Change applied and observed.** The reference engine reported
  `cache.cache_dtype = auto`, the candidate `fp8`, with no evidence gaps in
  either run. Model and runtime pins verified after launch.
- **Comparison VALID.** Only the five declared KV-cache paths differed.
- **Behavior.** Trajectory agreement 100/100: all 12 measured greedy
  trajectories were identical under the FP8 cache on this model.
- **Quality.** Task checks passed 6/12 in the reference and 6/12 in the
  candidate. The same two sentiment checks and the order-ID check fail in both.
- **Policy FAIL.** Both pass-rate criteria require 1.0 and observed 0.5.

The failure is attributable to the workload and model, not to FP8: the
reference fails identically. The profile reports that distinction instead of
attributing it to the change. Loaded-engine latency changed by −3.2%
(sequential calls after warmup; not a speedup claim).

[`run.log`](run.log) records preparation and verification (local paths replaced
with `${WORK}`). The recipe and run records keep their non-personal machine
paths unmodified so the bound record and evidence digests stay valid.
[`files-sha256.json`](files-sha256.json) lists the retained bytes.

## Not established

FP8 quality on other models, workloads, devices, attention backends or scaling
methods; production performance; or any universal threshold.
