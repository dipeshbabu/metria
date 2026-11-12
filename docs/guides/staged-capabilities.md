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
`token_top_logprobs` captures; the llama.cpp adapter reports the second as
`experimental`.

**Native provider.** Build the
[qualified capture patch](../../tools/qualification/README.md), then apply
`tools/qualification/staged/llamacpp-topk-capture.patch` on top. The staged
patch only writes distributions when `KV_FIDELITY_TOPK` is set, so the token
stream is otherwise unchanged. It is a separate file because qualified
providers are pinned to the SHA-256 of the unchanged capture patch.

**Remaining qualification:** compile the layered provider at the pinned
llama.cpp revision; retain a self-comparison (expected KL of exactly zero) and a
known KV-precision change; and confirm the captured top-k matches the logits
from an independent `llama-perplexity --kl-divergence` run on the same context.

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

**Remaining qualification:** retain native CPU and CUDA runs with the pinned
vLLM 0.30.0 wheels showing each setting applied, plus a negative run where the
engine rejects an unsupported value.
