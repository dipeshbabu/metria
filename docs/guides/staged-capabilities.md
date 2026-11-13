# Staged capabilities

Staged capabilities are implemented and unit-tested, but no retained native run
qualifies them yet. They exist so the qualification run has finished code to
exercise. Treat their output as engineering evidence, not as a supported
verification result.

Each staged result is labeled `"qualification": "staged"`. No policy target
consumes it, and the qualified `metria verify` profiles reject the inputs that
enable it. Promotion follows the
[evidence requirements](../architecture/verifier-product.md#evidence-requirements-for-extending-support):
a pinned native run, retained raw evidence, negative semantic tests, and a
copyable example.

## Trajectory KL divergence

Trajectory agreement reports where sampled tokens first differ. Trajectory KL
divergence reports how far the model's next-token distribution moved before
that point, which catches drift that has not yet changed a sampled token.

**Method** (`metria.trajectory_kld` 0.1.0):

- Each run records the full-vocabulary top-20 log probabilities at every decode
  step, taken from the raw logits before sampler transforms.
- At step *i* the reference and candidate condition on identical context only
  while their sampled prefixes agree. Each prompt is therefore scored over steps
  `0..first_divergence` (inclusive) that both runs produced. Later steps are
  excluded rather than compared across different contexts.
- Each step uses the existing `approximate_topk_kl` estimator:
  `KL(reference || candidate)` over the union of visible token IDs plus one
  omitted-mass bucket. It is an approximation and is labeled as one.
- `trajectory_kld_mean` pools all scored steps; `trajectory_kld_max` is the
  largest single step. Coverage is the fraction of prompts with at least one
  scored step.

**Use it** from Python with `metria.measurements.TokenDistributionProtocol` and
`TrajectoryDivergenceAnalysis`. The protocol requires the `token_ids` and
`token_top_logprobs` captures; the llama.cpp and vLLM adapters report the
second as `experimental`. The study lifecycle admits an experimental capture only when the
run opts in explicitly and its token-ID provider is qualified by SHA-256:

```json
"trial_policy": {"capability_overrides": ["metria.staged_captures"]}
```

**Native provider.** Build the
[qualified capture patch](../../tools/qualification/README.md), then apply
`tools/qualification/staged/llamacpp-topk-capture.patch` on top. The staged
patch only writes distributions when `KV_FIDELITY_TOPK` is set, so the token
stream is otherwise unchanged. It is a separate file because qualified
providers are pinned to the SHA-256 of the unchanged capture patch.

**Functional check.** The layered provider has been compiled at the pinned
revision (GCC 13.3, Ubuntu 24.04 under WSL2, no compiler warnings) and passes
`tools/qualification/staged/check_topk_provider.py` with the pinned
`stories260K.gguf` fixture:

- the token stream is identical to the qualified provider's with and without
  `KV_FIDELITY_TOPK`;
- every step carries a top-20 distribution accepted by Metria's strict reader;
- at temperature 0 the sampled token is the captured top-1 at every step;
- repeating an identical run through the llama.cpp adapter gives KL of exactly 0;
- enabling flash attention left every sampled token unchanged but moved the
  distributions by a mean of about 5e-7 nats, which token agreement cannot see.

This is a functional check, not qualification: nothing is retained, and the
fixture's 8-wide attention heads cannot load quantized (`q8_0`, block 32) KV
caches.

**Native evidence.** On Qwen2.5-0.5B-Instruct (64-wide heads), the
[retained runs](../../artifacts/qualification/trajectory-kld-qwen05b/README.md)
show a self-comparison KL of exactly 0 over 379 steps; f16 → `q8_0` KV at
0.038 nats and f16 → `q4_0` KV at 4.89 nats. An independent
`llama-perplexity --kl-divergence` cross-check over pinned wikitext-2 gives
0.015 and 1.98 nats: the same ranking and a consistent ratio.
`tools/qualification/staged/qualify_trajectory_kld.py` reproduces them.

**vLLM provider.** The vLLM adapter requests `logprobs=20` from `SamplingParams`
only when `token_top_logprobs` is requested, and refuses to capture unless the
engine's `logprobs_mode` is `raw_logprobs` (the default), so distributions are
the model's own, before temperature or penalties. Each step keeps exactly the
ranks 1–20; the sampled token vLLM appends when it falls outside the top 20 is
dropped. Every generated token must carry a distribution, or the run fails.
Invocations record `"top_logprobs": {"top_k": 20, "mode": "raw_logprobs",
"qualification": "staged"}`.

On the pinned vLLM `0.30.0+cpu` wheel with SmolLM2-135M, the
[retained CPU runs](../../artifacts/qualification/vllm-trajectory-kld-cpu/README.md)
show a fresh-engine repeat at KL exactly 0 over 64 steps, and float32 →
bfloat16 model weights at a mean of 0.056 nats (max 0.41) with token agreement
95.3. `tools/qualification/staged/observe_vllm_gpu.py run --distribution`
reproduces them; the same flag works with `--backend cuda`.

**Remaining before promotion:** the method stays staged until a verify profile
consumes it with an explicit policy target and it is retained on a second
model or build.

## vLLM staged treatments

The vLLM adapter's qualified treatment remains `vllm.kv_cache`. Two staged
`runtime_feature` treatments map to documented engine arguments:

| Treatment | Settings | Read back from |
| --- | --- | --- |
| `vllm.scheduler` | `enable_chunked_prefill` (bool), `max_num_batched_tokens` (positive int) | `scheduler_config` |
| `vllm.cache_layout` | `block_size` (8, 16, 32, 64 or 128) | `cache_config` |

Every setting follows requested → resolved → observed. After the engine is
constructed, each value is compared with the engine configuration: a
contradiction fails the launch, and a field the engine does not expose is
retained as `unobserved`. Observations include a `staged_treatments` block.

Preflight blocks these treatments through the built-in
`vllm.staged_treatments` capability check. To run one in a study, opt in
explicitly:

```json
"trial_policy": {"capability_overrides": ["vllm.staged_treatments"]}
```

**CUDA observation.** On an NVIDIA L4 with the pinned `0.30.0+cu129` wheel,
`enable_chunked_prefill=true`, `max_num_batched_tokens=2048` and `block_size=32`
were all read back from the engine as `applied`, and greedy trajectories matched
an untreated reference exactly. See the
[retained L4 evidence](../../artifacts/qualification/l4-native-gpu/README.md).

**CPU observation.** With the pinned `0.30.0+cpu` wheel, the same three
settings read back as `applied`, and a scheduler budget below the model length
without chunked prefill was rejected by the engine's configuration validation at
launch. See the
[retained CPU evidence](../../artifacts/qualification/vllm-staged-treatments-cpu/README.md).

**vLLM provider.** The vLLM adapter requests `logprobs=20` from `SamplingParams`
only when `token_top_logprobs` is requested, and refuses to capture unless the
engine's `logprobs_mode` is `raw_logprobs` (the default), so distributions are
the model's own, before temperature or penalties. Each step keeps exactly the
ranks 1–20; the sampled token vLLM appends when it falls outside the top 20 is
dropped. Every generated token must carry a distribution, or the run fails.
Invocations record `"top_logprobs": {"top_k": 20, "mode": "raw_logprobs",
"qualification": "staged"}`.

On the pinned vLLM `0.30.0+cpu` wheel with SmolLM2-135M, the
[retained CPU runs](../../artifacts/qualification/vllm-trajectory-kld-cpu/README.md)
show a fresh-engine repeat at KL exactly 0 over 64 steps, and float32 →
bfloat16 model weights at a mean of 0.056 nats (max 0.41) with token agreement
95.3. `tools/qualification/staged/observe_vllm_gpu.py run --distribution`
reproduces them; the same flag works with `--backend cuda`.

**Remaining before promotion:** a verify profile that accepts these treatments
with task checks and an acceptance policy. The native CPU, CUDA and
engine-rejection runs it would rely on are retained.
