# Verify an FP8 KV-cache change (vLLM, CUDA)

This profile (`local_vllm_kv_cache_precision.v1`) compares a reference that uses
the model's automatic KV-cache dtype with a candidate that uses an FP8 cache
(`fp8`, `fp8_e4m3` or `fp8_e5m2`). Everything else is identical: model files,
tokenizer, runtime settings, workload, generation and trial schedule.

FP8 is a lossy change, so this profile always requires declared task checks on
the workload and an explicit [acceptance policy](verification-policies.md). A
completed comparison therefore ends in **PASS** or **FAIL**, never just
`VERIFIED`.

## Requirements

- Linux with an NVIDIA GPU of compute capability 8.9 or newer (Ada, Hopper or
  later). Older GPUs cannot run the pinned engine's FP8 cache path; see the
  [FP8 qualification status](fp8-qualification-status.md).
- The pinned `vllm==0.30.0+cu129` wheel with PyTorch `2.13.0+cu129`, prepared as
  for [prefix-cache verification](vllm-prefix-verification.md).
- A local model directory with a trusted file-to-SHA-256 descriptor.
- A JSONL workload whose prompts declare task checks, and a policy JSON file.

## Prepare and verify

```bash
metria recipe prepare-vllm \
  --model /path/to/local/model \
  --descriptor trusted-model-files.json \
  --workload tasks.jsonl \
  --policy policy.json \
  --kv-cache-dtype fp8 \
  --output fp8-study.json

metria verify fp8-study.json --output fp8-verification --json
```

Preparation refuses a recipe without a policy or without task checks.

## What the verifier checks

- **Requested → resolved → observed.** Each run requests one `vllm.kv_cache`
  treatment. The resolved KV-cache dtype and the engine's own
  `cache.cache_dtype` (from both the applied-state and identity readbacks) must
  equal the request. An engine that reports a different dtype is rejected before
  measurement (`EXECUTION_FAILED`); one that does not report it at all leaves an
  evidence gap (`INSUFFICIENT_EVIDENCE`).
- **Only the cache dtype differs.** The comparison may vary exactly five paths:
  `treatments`, `resolved.kv_cache.dtype`, `observed.configured.kv_cache.dtype`,
  `observed.applied.fields.cache.cache_dtype` and
  `observed.identity.applied.fields.cache.cache_dtype`. Prefix caching is off in
  both runs.
- **Pinned content.** Model/tokenizer files and the runtime installation are
  verified after launch, as in the prefix-cache profile.
- **Decision.** Trajectory agreement, task-check pass rates, reference
  repeatability and loaded-engine latency feed the policy. Pass rates for both
  runs are reported, so a workload the model cannot solve fails visibly in the
  reference as well as the candidate.

## Retained evidence

[`artifacts/qualification/fp8-kv-cache-verify-l4/`](../../artifacts/qualification/fp8-kv-cache-verify-l4/README.md)
is a complete run on an NVIDIA L4 with SmolLM2-360M-Instruct and the public
representative pilot workload. FP8 applied, trajectories were identical, and the
policy failed because the reference and candidate each passed 6 of 12 checks.
Separate [L4 observations](../../artifacts/qualification/l4-native-gpu/README.md)
show FP8 changing SmolLM2-135M trajectories substantially (agreement 48.4).

## Limits

Qualification covers the pinned wheel on observed SM89 hardware. It is not a
quality claim for FP8 on other models, attention backends, scaling methods or
devices; the result applies to your workload and criteria only.
