# Verify a tensor or pipeline parallel change (staged)

`local_vllm_parallelism.v1` compares a single-device **reference** with a
**candidate** that splits the same model across devices with tensor and/or
pipeline parallelism. Everything else is identical. Parallelism changes the
order of floating-point reductions, so outputs can drift even though the model
is the same; this profile measures that drift against your criteria.

!!! warning "Staged"
    No native multi-GPU run has qualified this profile yet
    ([#218](https://github.com/dipeshbabu/metria/issues/218)). It accepts only
    the pinned CUDA wheel (`vllm==0.30.0+cu129`).

## Prepare and verify

```bash
metria recipe prepare-vllm \
  --model /path/to/local/model \
  --descriptor trusted-model-files.json \
  --workload tasks.jsonl \
  --policy policy.json \
  --candidate-tensor-parallel-size 2 \
  --output parallel-study.json

metria verify parallel-study.json --output parallel-verification
```

Use `--candidate-pipeline-parallel-size N` instead of, or together with, the
tensor option. The candidate may span 2–8 devices in total.

## What the verifier checks

- **Only the parallel sizes differ.** For each size that changes, the comparison
  varies exactly its request, resolved setting, configured value and both
  engine readbacks (`observed.applied.fields.parallel.*` and the identity copy).
  Prefix caching is off in both runs.
- **The engine applied the request.** vLLM's own `parallel_config` must report
  the requested sizes. An engine that reports different sizes is rejected
  before measurement; one that does not report them leaves an evidence gap.
- **Enough devices.** Runs record every visible CUDA device (name, compute
  capability, memory, hashed UUID) under `observed.hardware.devices`. A
  candidate host with fewer devices than requested is an evidence gap.
- **A decision.** Task checks and an explicit policy are required, as for FP8
  KV caches. `--staged-kld` adds trajectory KL divergence.

## Status

The adapter side was exercised natively with the pinned `0.30.0+cpu` wheel,
whose CPU backend runs tensor-parallel ranks on separate core groups, using
SmolLM2-135M (9 attention heads, 3 KV heads) through Metria's real adapter:

- tensor parallel 1 and 3 both completed, and vLLM's `parallel_config` read
  back `tensor_parallel_size` 1 and 3 (and `pipeline_parallel_size` 1);
- greedy trajectories at tensor parallel 3 matched tensor parallel 1 exactly
  (agreement 100) on the four standard prompts;
- tensor parallel 2 was rejected by vLLM's configuration validation at launch,
  consistent with 9 heads not dividing by 2; the run failed before measurement
  rather than silently using another size.

This is a functional check of the adapter, not qualification: the profile
accepts only the CUDA wheel, and a two-GPU run is still required before
promotion.
