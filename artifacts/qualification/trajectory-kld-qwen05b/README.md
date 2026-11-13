# Native trajectory KL divergence on Qwen2.5-0.5B (staged)

On 2026-11-13 the staged top-k llama.cpp provider (`llama-completion` SHA-256
`b2f32cf9…52e8`) ran through Metria's real llama.cpp adapter and study lifecycle
on Linux (WSL2, x86_64). Pinned build: llama.cpp `434ddbb` with the qualified
capture patch plus `tools/qualification/staged/llamacpp-topk-capture.patch`, CPU
only, GCC 13.3. The model is `Qwen/Qwen2.5-0.5B-Instruct-GGUF` at revision
`9217f5db79a29953eb74d5343926648285ec7e67`, file
`qwen2.5-0.5b-instruct-q8_0.gguf` (SHA-256 `ca59ca7f…844e`, matching the
upstream LFS hash). Its attention heads are 64 wide, so quantized KV caches load.

Every run used flash attention, greedy decoding, context 512, 64 generated tokens
and six plain-completion prompts, with the token-ID provider qualified by its
SHA-256 and the staged capture admitted through the explicit
`metria.staged_captures` override. Local paths are replaced with `${MODEL_DIR}`
and `${STAGED_BUILD}`.

| Comparison | Trajectory KLD (mean / max, nats) | Corpus KLD (nats) | Trajectory agreement |
| --- | --- | --- | --- |
| f16 vs f16 repeat | **0.0** / 0.0 over 379 steps | — | 100 |
| f16 → `q8_0` KV | 0.0381 / 0.535 over 121 steps | 0.0154 | 30.6 |
| f16 → `q4_0` KV | 4.89 / 26.0 over 14 steps | 1.98 | 2.1 |

- **Self-comparison:** identical runs give KL of exactly zero, so the capture
  and estimator add no noise of their own.
- **Independent cross-check:** corpus KLD comes from
  `llama-perplexity --kl-divergence` against an f16-KV base over 8 × 512-token
  chunks of the pinned wikitext-2 test split (SHA-256 `173c87a5…dd08`). Both
  estimators rank `q4_0` far above `q8_0`, and the trajectory value is about
  2.5× the corpus value for both changes. Trajectory KLD scores generation from
  short prompts only up to the first divergent token, so its magnitude is not
  expected to equal the corpus figure.
- **Effect size:** a `q4_0` KV cache breaks this model's generation (two
  prompts diverge at the first token); `q8_0` drifts measurably while keeping
  one prompt identical for all 64 tokens.

[`summary.json`](summary.json) holds every figure above; the four run records
retain the full per-step top-20 distributions they were computed from.
[`files-sha256.json`](files-sha256.json) lists the retained file bytes.

## Not established

This retains native evidence for the staged method on one model, build and
host. The method stays staged: no policy target consumes it and no verify
profile uses it. It is not a model-quality claim for these KV cache types.
