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

A suitable GPU will be supplied for the remaining qualification. Before promoting
the [staged recipe](../../examples/verification/staged/kv-precision.json), retain
successful reference/candidate runs with immutable model/tokenizer/runtime pins,
authoritative applied cache-format evidence, native output captures, compatible
measurements and explicit task-quality criteria. A mocked test or device name
alone cannot supply that evidence. Track the remaining work in
[issue #161](https://github.com/dipeshbabu/metria/issues/161).
