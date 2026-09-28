# Verify a vLLM prefix-cache change

The Metria 0.2 development line supports a local reference with prefix caching
disabled and a candidate with it enabled. Both use the same pinned model,
tokenizer, installed vLLM content, generation settings, workload, and device.
This is a sequential offline comparison with greedy plain completion, one device,
and eager execution. Runtime upgrades, FP8 KV precision, remote endpoints, chat
templates, tensor parallelism and arbitrary treatments are separate scopes.

## Prepare a controlled recipe

Use a qualified vLLM `0.30.0+cpu` or `0.30.0+cu129` environment. The
[runtime qualification guide](vllm-qualification.md) records the installed wheel
identities, hardware and custom dependency constraints. Install the Metria wheel
inside that environment; base Metria does not install an engine stack.

Supply an existing local model directory and an independently trusted file manifest.
The [SmolLM2 fixture descriptor](../../tools/qualification/vllm-smollm2-135m.json)
pins model and tokenizer payloads for the retained integration example. No weights
are bundled or automatically downloaded. Store recipes outside the model directory.

Run the preparation tool from the same runtime environment:

```bash
python tools/qualification/prepare_vllm_prefix_verification.py \
  --model /path/to/pinned/model \
  --descriptor tools/qualification/vllm-smollm2-135m.json \
  --workload examples/verification/vllm-prefix-workload.jsonl \
  --output prefix-study.json

metria verify prefix-study.json --output prefix-verification
```

The tool verifies model/tokenizer bytes, fingerprints the installed vLLM wheel
payload and dependency versions, and writes explicit settings and comparison paths.
Inspect that recipe before using it. The current allowed change is exactly caching
off to on; unrelated changes or comparison waivers are rejected before launch.
User workloads use JSONL `id`, `prompt`, optional `category` and `generation` fields.

## What is measured

Each engine remains loaded for its workload. Prompts execute sequentially in their
declared order. The default is one warmup trial followed by three measured trials.
The public prefix-cache reset must return success before every trial. Warmups retain
evidence but contribute neither latency samples nor trajectory comparisons.

Native output token IDs, prompt-token counts, cache-hit counts and completion state
are retained. Verification requires no cache reuse in the reference and an observed
candidate cache hit in the measured repeated-prefix workload. A missing counter
remains unknown; it is never converted to zero. Runtime-applied caching and context
must match the request. Local model/tokenizer files are checked against their pins
before launch and from the runtime-reported source after launch.

The latency method measures the `session.infer` call with `perf_counter_ns` after
engine startup. It includes prompt evaluation and generation, while excluding
engine loading, warmup and cache reset. It reports complete sample means and a
compatible reference/candidate delta. This is not TTFT, decode-only throughput,
serving throughput, peak GPU/KV memory or a statistical speedup guarantee.
Trajectory counts refer to prompt/trial captures, so the two-prompt example with
three measured trials retains six comparisons per role.

## Failure and cleanup

Native work runs in a bounded child process using the shared process-tree lifecycle.
The supervisor configures engine worker spawning and offline model access. Its
deadline covers both runs, including startup and cleanup; the default is 900 seconds.
Raw worker output is bounded and represented by hashes and truncation flags.

The supervisor validates worker receipt and run-record digests before publishing
`verification.json`. Completed records survive later failure. A missing run becomes
failed/timed-out/interrupted or cancelled evidence, with no behavioral analysis or
policy PASS. Output directories must be new. The [standard exit codes](verification-ci.md)
remain unchanged, including 5 for execution failure and 130 for interruption.

`VERIFIED` means the scoped comparison completed with the required evidence. It does
not assert model quality, universal determinism or deployment acceptance. Use your
own workload and explicit acceptance policy for an engineering decision.

## Retained qualification

[CPU and CUDA evidence](../../artifacts/qualification/vllm-prefix-cache/README.md)
was captured from installed wheels on 2026-09-28. Both profiles completed and
demonstrated the cache change. These tiny-model runs qualify the interface and
evidence contract; their timings are not production benchmark claims.
