# Reproduce the pinned vLLM qualification

This qualification exercises Metria's real vLLM adapter with SmolLM2-135M,
plain greedy completion, two eight-token prompts, reset, repeated capture,
observation, and cleanup. It is an integration check, not a performance or
model-quality claim. The CLI verifier remains scoped to llama.cpp CPU thread
changes; this evidence qualifies the Python adapter paths named below.

The [retained CPU and GTX 1650 runs](../../artifacts/qualification/vllm-0.30.0/README.md)
include their exact environment, observed evidence, reset results, and limits.

The runtime is vLLM 0.30.0, using separate official CPU and CUDA 12.9 wheels.
The [model descriptor](../../tools/qualification/vllm-smollm2-135m.json) pins the
model/tokenizer revision and hashes every input file. The qualifier rechecks
the wheel and model files before execution. No model weights are committed.

## Environments

Use separate Linux environments for CPU and CUDA. The wheel names are:

```text
vllm-0.30.0+cpu-cp38-abi3-manylinux_2_39_x86_64.whl
vllm-0.30.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
```

Download from the [official v0.30.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.30.0).
The qualifier contains the release asset SHA-256 for each wheel. Install with
the corresponding PyTorch CPU or CUDA 12.9 index, using Python 3.12 and PyTorch
2.13.0. Keep the environment isolated from the repository's default workspace.

These retained environments explicitly override upstream's obsolete setuptools
constraint with `setuptools==84.0.0` to meet Metria's security minimum. Record that
deviation; do not describe the environment as an unmodified upstream dependency
installation. The result retains all installed distribution versions. Reproduce
those versions when repeating a retained qualification, and treat any changes as
a new qualification. The override concerns a build-tool constraint, and the real
engine/capture tests determine whether this scoped environment works.

The CPU lane uses float32 with two OpenMP threads and 1 GiB of KV cache. The CUDA
lane uses float16, one sequence, a 128-token model limit, eager execution, and
65% GPU-memory utilization. The retained GPU result applies only to its observed
device and driver. Device visibility alone does not qualify an engine.

## Execute

Download only the descriptor's files at its exact upstream revision using
Hugging Face's snapshot downloader. Pass the local snapshot directory. Do not
substitute a mutable branch or a similarly named tokenizer.

From a checkout of the desired Metria revision, using the runtime environment's
Python:

```bash
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
python tools/qualification/qualify_vllm.py \
  --backend cuda --model-root /path/to/pinned/snapshot \
  --runtime-wheel /path/to/pinned/vllm-wheel.whl \
  --source-revision "$(git rev-parse HEAD)" --output qualification-output
```

For CPU, select `--backend cpu`, set `VLLM_CPU_KVCACHE_SPACE=1` and
`VLLM_CPU_OMP_THREADS_BIND=0-1`, and preload the environment's `lib/libiomp5.so`.
Use a new output directory. Failed execution retains a failed run record; it must
not be advertised as qualification. A successful record includes observed identity
authority states. Unknown chat-template metadata and partial applied introspection
remain explicitly unknown/partial; plain completion does not establish chat support.

The output is `qualification.run.json` plus `qualification.json`. Local model,
cache, runtime-environment, and source paths are replaced consistently with named
placeholders before the public record is serialized. The normalized record's hash
identifies that public representation, not the original private paths.

## Controlled runner and staleness

[Manual runtime qualification](../../.github/workflows/runtime-qualification.yml)
runs only from `main` on a maintainer-controlled Linux runner with label
`metria-qualification`. Configure `METRIA_VLLM_CPU_PYTHON`,
`METRIA_VLLM_CUDA_PYTHON`, the corresponding `*_WHEEL` variables, and
`METRIA_VLLM_MODEL_ROOT` to the preinstalled pinned environments and input cache.
The job performs no unpinned model download. Maintainers own provisioning and
manual dispatch; ordinary PR-required fixture tests remain independent of hardware.

Qualification becomes stale when the adapter, runtime/model/tokenizer hashes,
environment packages, driver, hardware, or recipe changes. Rerun the affected
lane before extending a support claim. Review retained qualifications at least
every 90 days even when pins are unchanged. Failed/stale lanes do not inherit
success from mocked tests or another device's results.
