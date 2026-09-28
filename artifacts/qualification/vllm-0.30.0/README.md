# Pinned vLLM 0.30.0 qualification

Both retained runs completed real inference, reset, repeated greedy token capture,
observation, and cleanup on 2026-09-28 UTC. They use SmolLM2-135M at revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, with verified model/tokenizer file hashes.

| Lane | Environment | Evidence |
|---|---|---|
| CPU | WSL Linux x86_64, float32, PyTorch 2.13 CPU, two OpenMP threads | [Summary](cpu/qualification.json), [run record](cpu/qualification.run.json) |
| CUDA | WSL, NVIDIA GeForce GTX 1650 4 GiB, compute capability 7.5, driver 566.36, PyTorch 2.13 / CUDA 12.9, float16 | [Summary](cuda/qualification.json), [run record](cuda/qualification.run.json) |

Each summary retains all installed package versions, the official runtime-wheel
hash, input hashes, executed Metria source revision, hardware/software evidence,
and the explicit setuptools 84.0.0 override of upstream's obsolete constraint.
This is a qualified custom dependency environment, not an unmodified upstream
install. The model/runtime versions alone are not enough to reproduce it.

The source revision identifies the executed adapter and qualifier. Local paths
were consistently replaced with placeholders in the public records; a shared
Hugging Face blob-cache path was additionally normalized during curation. The
summary's record hash identifies the curated public bytes. No numerical result,
captured token ID, identity authority state, or lifecycle event was changed.

Observed model/tokenizer/runtime identity and exposed configuration were checked
before prompts ran. Unavailable chat-template metadata and incomplete applied
introspection remain unknown/partial. These plain-completion runs do not qualify
chat templates, quantization/KV treatments, different models, other GPUs, runtime
upgrades, task quality, or performance. The CPU and GPU outputs are separate
qualification runs; they are not presented as a controlled performance comparison.

The CUDA engine's own shutdown manager force-terminated its remaining engine
process after its graceful-shutdown timeout. Metria's public shutdown call returned
and the process exited; the retained success does not imply every internal engine
worker used graceful teardown.

See the [reproduction procedure](../../../docs/guides/vllm-qualification.md)
and [manual workflow](../../../.github/workflows/runtime-qualification.yml).
Rerun on any relevant source, input, dependency, driver, device, or recipe change;
review unchanged pins after 90 days. This evidence does not expand the public
`metria verify` CLI beyond its documented llama.cpp scope.
