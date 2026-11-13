# Native vLLM CPU trajectory KL divergence (staged)

On 2026-11-13 the pinned vLLM `0.30.0+cpu` wheel (SHA-256 `0ee75278…ed78`) ran
through Metria's real vLLM adapter and study lifecycle on Linux (WSL2, x86_64),
with the same environment as the
[staged-treatment CPU observations](../vllm-staged-treatments-cpu/README.md):
two OpenMP threads, 1 GiB of KV cache, eager execution, one sequence, a
128-token model limit, greedy decoding and 16 new tokens for each of four
prompts. The model is the pinned SmolLM2-135M at revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`; every file was verified against its
descriptor hash before each run. Local paths are redacted.

Each run measures `metria.decode_time_top_logprobs` (`TokenDistributionProtocol`)
and opts in with `"capability_overrides": ["metria.staged_captures"]`. The
adapter reported the capture as `experimental` in preflight, and every
invocation records `"top_logprobs": {"top_k": 20, "mode": "raw_logprobs",
"qualification": "staged"}`.

| Candidate vs `reference` (float32) | Trajectory KL mean (nats) | Max (nats) | Scored steps | Token agreement |
| --- | --- | --- | --- | --- |
| `repeat` (identical, fresh engine) | 0 | 0 | 64 | 100 |
| `bf16` (model dtype bfloat16, read back as configured) | 0.0559 | 0.406 | 62 | 95.3 |

The repeat shows the capture is deterministic across engine launches; the
bfloat16 change moves distributions measurably and diverges one prompt at step
13. Scores come from `compare-reference-*.json`; `files-sha256.json` hashes
every retained file.

Reproduce (Linux, `src` on `PYTHONPATH`, the vLLM CPU environment from
`tools/qualification/README.md`):

```bash
for v in reference repeat bf16; do
  python tools/qualification/staged/observe_vllm_gpu.py run --distribution \
    --backend cpu --variant $v --model-root "$MODEL" \
    --descriptor tools/qualification/vllm-smollm2-135m.json \
    --runtime-wheel "$WHEEL" --output out
done
python tools/qualification/staged/observe_vllm_gpu.py compare --candidate bf16 --output out
```

This is engineering evidence for a staged capability. It does not promote
trajectory KL divergence or qualify the vLLM capture.
