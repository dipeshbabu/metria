# FP8 KV-cache qualification status

Positive FP8 verification remains staged. The existing GTX 1650 has compute
capability 7.5. A real local run of the pinned vLLM `0.30.0+cu129` wheel reached
its Triton attention backend and rejected FP8 KV cache because that path requires
SM89 or newer. This was a native capability failure, not a timeout or a missing
model download.

The [retained evidence](../../artifacts/qualification/fp8-unsupported/preflight.json)
records the installed preflight result, hardware, source revision and wheel hash.
The [native failure log](../../artifacts/qualification/fp8-unsupported/native-stderr.log)
and [content index](../../artifacts/qualification/fp8-unsupported/sha256.json)
preserve the earlier engine attempt. These are negative capability observations;
they do not qualify an FP8 comparison.

The first-party adapter now detects that exact observed runtime/device
combination during preflight and returns an actionable unsupported result before
constructing the inference engine. Ordinary automatic-cache workflows continue
to work. The guard does not infer positive support for other GPUs, wheel builds,
attention backends or cache formats.

## Positive native observation on SM89

On 2026-11-13 an NVIDIA L4 (compute capability 8.9, driver 595.84) ran the same
pinned wheel through Metria's real vLLM adapter with the pinned SmolLM2-135M
files. The [retained L4 evidence](../../artifacts/qualification/l4-native-gpu/README.md)
shows:

- the candidate requested `fp8`, resolved `kv_cache_dtype=fp8`, and the
  constructed engine reported `cache.cache_dtype = fp8`; the reference reported
  `auto`;
- greedy trajectories changed: agreement 48.4/100 over four prompts, with two
  prompts diverging at the first generated token and one matching fully;
- the standard adapter qualification also completed on this device.

This satisfies the immutable pins, authoritative applied cache-format evidence,
native output captures and compatible measurements listed below. It is retained
engineering evidence for one model, device and wheel, not a general FP8 quality
claim.

## Remaining before promotion

Promoting the [staged recipe](../../examples/verification/staged/kv-precision.json)
to a `metria verify` profile still requires an FP8 verification route with
predeclared task checks and an explicit acceptance policy, run against this kind
of evidence. A mocked test or device name alone cannot supply that evidence.
`tools/qualification/staged/setup_vllm_cuda.sh` and `observe_vllm_gpu.py`
reproduce the L4 observations on another SM89+ host. Track the remaining work in
[issue #161](https://github.com/dipeshbabu/metria/issues/161).
