# Choose a verification workflow

Start with the change you need to evaluate. A workflow qualifies a specific
comparison contract; support for an engine does not qualify every possible
configuration of that engine.

## Local llama.cpp

| Change | Workflow |
| --- | --- |
| CPU thread count with a fixed model and provider | [CPU thread verification](metria-verify.md) |
| Independently pinned CPU runtime builds | [Build verification](llamacpp-build-verification.md) |
| A controlled GGUF Q8_0 weight conversion | [GGUF quantization verification](gguf-quantization-verification.md) |

These workflows retain qualified capture evidence and bind it to the pinned
model and runtime. Read each guide's preparation requirements before execution.

## Local vLLM

| Change | Workflow |
| --- | --- |
| Prefix caching off versus on | [Prefix-cache verification](vllm-prefix-verification.md) |
| A CPU runtime upgrade in separate environments | [Runtime-upgrade verification](runtime-upgrade-verification.md) |
| Local streaming, throughput, and native allocation behavior | [Serving measurements](serving-measurements.md) |
| Automatic versus FP8 KV cache (CUDA, SM89+) | [FP8 KV-cache verification](fp8-kv-cache-verification.md) |
| Single device versus tensor/pipeline parallel (CUDA, staged) | [Parallelism verification](parallelism-verification.md) |

Use the [runtime qualification guide](runtime-qualification.md) to understand
the retained evidence for a platform. Staged options, such as trajectory KL
divergence and the vLLM scheduler and cache-layout treatments, have their own
boundaries; check the [staged capability notes](staged-capabilities.md) before
treating a result as a qualified end-to-end workflow.

## Running servers (staged)

| Change | Workflow |
| --- | --- |
| A declared change between two running OpenAI-compatible servers | [Remote endpoint verification](endpoint-verification.md) |

Server identity is claimed rather than verified, so this workflow always needs
a policy and records the limitation in every report.

## Decide what acceptance means

Behavioral drift, task outcomes, and systems impact answer different questions:

- [Task checks and repeatability](verification-decisions.md) separate token
  differences from declared output requirements.
- [Policies](verification-policies.md) apply your own numeric or exact criteria.
- [Performance measurements](verification-performance.md) state their timing
  boundary and method before comparing latency.
- [Repeated trials](verification-trials.md) retain warmup, isolation, and trial
  identity rather than combining unrelated measurements.

For a first look at the output, run the [synthetic report walkthrough](first-report.md).
For automation, use the [CI guide](verification-ci.md) and the
[copyable workflow examples](../../examples/verification/README.md).
