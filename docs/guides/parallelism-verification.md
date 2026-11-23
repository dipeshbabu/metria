# Verify a tensor or pipeline parallel change (staged)

`local_vllm_parallelism.v1` compares a single-device **reference** with a
**candidate** that splits the same model across devices with tensor and/or
pipeline parallelism. Everything else is identical. Parallelism changes the
order of floating-point reductions, so outputs can drift even though the model
is the same; this profile measures that drift against your criteria.

!!! warning "Staged"
    A native 2×L4 run is retained (see [Status](#status)); the profile stays
    staged pending a second model or host. It accepts only the pinned CUDA
    wheel (`vllm==0.30.0+cu129`).

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

## NCCL on PCIe-only hosts

If tensor-parallel startup hangs right after vLLM reports its NCCL version,
the GPUs are probably connected only through a PCIe host bridge (`PHB` in
`nvidia-smi topo -m`). Export `NCCL_P2P_DISABLE=1` before `metria verify`; the
isolated worker inherits it, and the run records it under
`observed.numerics_environment`. (The retained 2×L4 run predates that field,
so its README states the setting instead.)

## Status

On two NVIDIA L4 GPUs with Qwen2.5-0.5B-Instruct (14 heads),
[`parallelism-l4`](../../artifacts/qualification/parallelism-l4/README.md)
retains both cases with engine readback of the requested sizes and both GPUs
recorded:

| Candidate | Verdict | Agreement | Trajectory KLD (nats) |
| --- | --- | --- | --- |
| tensor parallel 2 | FAIL (agreement bound 0.9) | 0.891 | 0.0077 |
| pipeline parallel 2 | PASS | 1.0 | 0 |

Tensor parallelism moved distributions very little but enough to change some
greedy tokens; pipeline parallelism changed nothing. Both ran with
`NCCL_P2P_DISABLE=1`.

Earlier, the adapter side was exercised natively with the pinned `0.30.0+cpu` wheel,
whose CPU backend runs tensor-parallel ranks on separate core groups, using
SmolLM2-135M (9 attention heads, 3 KV heads) through Metria's real adapter:

- tensor parallel 1 and 3 both completed, and vLLM's `parallel_config` read
  back `tensor_parallel_size` 1 and 3 (and `pipeline_parallel_size` 1);
- greedy trajectories at tensor parallel 3 matched tensor parallel 1 exactly
  (agreement 100) on the four standard prompts;
- tensor parallel 2 was rejected by vLLM's configuration validation at launch,
  consistent with 9 heads not dividing by 2; the run failed before measurement
  rather than silently using another size.

That CPU run was a functional check of the adapter; the 2×L4 run above is
the profile's native evidence.
