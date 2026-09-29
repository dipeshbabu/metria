# Verify a GGUF weight-quantization change

The Metria 0.2 CLI prepares a CPU comparison between an immutable
F32/F16 GGUF and a Q8_0 conversion made with a pinned native `llama-quantize`.
Both runs use the same qualified capture provider, workload, generation settings
and thread counts. This scope covers single-file, little-endian GGUF version 3.

Prepare two new output paths and a JSONL workload with `id`, `prompt` and optional
[task checks](verification-decisions.md):

```bash
metria recipe prepare-gguf-quantization \
  --bin-dir /path/to/qualified/bin \
  --model /path/to/source.gguf \
  --model-sha256 TRUSTED_SOURCE_SHA256 \
  --candidate-model /path/to/new-q8-model.gguf \
  --workload prompts.jsonl \
  --policy acceptance-policy.json \
  --output quantization-study.json

metria verify quantization-study.json --output quantization-verification
```

The binary directory must contain the patched `llama-completion` provider and
`llama-quantize`. Supply `--quantizer` for a different tool path and optionally
`--quantizer-sha256` to check an independently trusted tool pin. The
[provider build instructions](../../tools/qualification/README.md) identify the
native source and capture patch. The base Metria package downloads no models or
inference engines.

Preparation verifies the source content, hashes the quantizer, runs conversion
with a deadline, and inspects the actual tensor table. It verifies that embedded
tokenizer metadata, nonquantization model metadata, and tensor names/shapes were
preserved. It then probes both models with the native capture provider before
writing the recipe. Conversion and qualification receipts are saved beside it.
Failed native conversions retain a receipt; failed probes retain their records.

Candidate weights are written in a private staging directory and published with
an exclusive hard link. Existing output files are never overwritten. Use a
filesystem that supports hard links. The tool and source pins are checked again
after conversion; a changed artifact cannot silently become a trusted recipe.

Q8_0 is the requested conversion format, not a promise that every tensor is stored
with that type. Native conversion may keep F32 normalization tensors or use F16
fallbacks for incompatible tensor widths. Reports include the observed inventory
of F32, F16 and Q8_0 tensors. A filename or a `general.file_type` label alone cannot
establish quantization. Missing optional labels remain missing.

Inspection reads at most 64 MiB of metadata and tensor-table data and rejects
unsupported storage types, split models, duplicate names, oversized/nested
metadata, and invalid or overlapping tensor regions. Whole-file hashes bind the
weights; the header inspection does not load them into Python. The verifier also
requires native token capture and vocabulary readback matching the embedded
tokenizer. Storage fingerprints describe the GGUF file, not activation precision
or device-memory usage.

Interpret drift, declared task-check outcomes and compatible cold-process latency
separately. A policy PASS applies only to your checks and workload. The default
comparison alone cannot establish answer quality, useful memory savings at
runtime, or a production performance improvement.

The [retained installed-wheel qualification](../../artifacts/qualification/gguf-q8-quantization/README.md)
includes the actual conversion receipt, both probes, run records and a policy
declared before execution. It passed on the tiny pinned workload; its scope and
limitations are recorded alongside the original evidence.
