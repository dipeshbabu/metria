# Installed-wheel GGUF Q8_0 qualification

On 2026-09-28, an installed Metria wheel converted a pinned stories260K GGUF with
the pinned native quantizer, qualified both model/provider combinations, and
completed verification. The policy was declared before execution: every candidate
completion must be nonempty and sampled trajectory agreement must be at least
0.99. The result was `PASS`, with all three presence checks passing and sampled
trajectory agreement of 1.0 on this workload.

- Metria source: `3eedaaa75ff6b5c33b27ebef7afbeb96c364cabb`.
- Installed wheel SHA256: `fbe8e51646a63f471216f0ddfb3bff9606bf921ac724e44ced01db26f98dc832`.
- Native llama.cpp source: `434ddbbc0e30522e897670681e503b797c12b7c1`, using the
  retained capture patch and Release/O2 build from the build-comparison qualification.
- Quantizer SHA256: `7b890eb928d55e96fb2b4d4b5e9a456be853f924de7f9d61a0ad7fab83359c91`.
- Provider SHA256: `0fe6d5edf465771841c16dcdb52119a61c1a9d7300655f1ca8701dcd11141f42`.
- Source GGUF SHA256: `270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d`.
- Candidate GGUF SHA256: `ff8701a45da28293b7af11a695ab33575b692e9b109a2eae6238c906c31415da`.

Actual storage changed from 48 F32 tensors to 32 Q8_0, 11 F32 and 5 F16 tensors.
The tokenizer, nonquantization metadata and tensor-layout fingerprints matched.
The candidate file is 379,168 bytes. Neither this size nor the storage inventory
is a measurement of inference-time memory usage.

[qualification.json](qualification.json) records the identities and limitations.
[study.conversion.json](study.conversion.json) retains the actual native conversion
receipt, including its fallback warning. The actual [recipe](study.json),
[workload](workload.jsonl), [policy](policy.json), independent
[reference](study.reference.qualification.run.json) and
[candidate](study.candidate.qualification.run.json) probes, and completed
[report](verification/report.md) and [canonical result](verification/verification.json)
are retained. [files-sha256.json](files-sha256.json) binds 18 original evidence files
copied without rewriting their bytes. Model weights remain outside the repository;
the [installed preparation guide](../../../docs/guides/gguf-quantization-verification.md)
explains how to produce a fresh pinned candidate.

This is a tiny-model maintainer contract test. Presence checks do not establish
answer correctness, three prompts do not qualify arbitrary workloads, and the
cold-process timings do not establish a statistical speedup. This is not an
external user pilot or a general recommendation for Q8_0 deployment.
