# vLLM prefix-cache verification qualification

These original installed-wheel results were captured on 2026-09-28 using source
commit `01ae0621368e98647f6bf48ff1af82110ee50a4a` and the pinned SmolLM2-135M
model/tokenizer revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.

| Environment | Outcome | Evidence |
|---|---|---|
| vLLM 0.30.0+cpu, PyTorch 2.13 CPU, Ubuntu WSL on i7-9750H | VERIFIED; both runs completed, no comparison issues/evidence gaps | [Report](cpu/report.md), [result](cpu/verification.json), [environment](cpu/environment.json) |
| vLLM 0.30.0+cu129, PyTorch 2.13 CUDA12.9, GTX1650 4GiB/SM7.5 | VERIFIED; both runs completed, no comparison issues/evidence gaps | [Report](cuda/report.md), [result](cuda/verification.json), [environment](cuda/environment.json) |

Each case used two synthetic repeated-prefix prompts, one excluded warmup trial,
three measured trials, and a confirmed prefix-cache reset before each trial.
The reference disabled caching; the candidate enabled it and showed native cache
hits. Both cases retained matching immutable model/tokenizer/runtime identities,
native output token trajectories, and six measured request-latency samples per role.

`study.json`, `reference.run.json`, `candidate.run.json`, `verification.json`, its
`manifest.json` compatibility alias, and `report.md` retain their original bytes.
The model was staged at a non-personal `/var/tmp` path before execution; no result
normalization or path redaction was needed. `qualification.json` records the source
revision, installed Metria wheel hash, runtime version and controlled environment.
The installed environments retain the documented setuptools84 override needed to
meet the repository security floor; they are custom dependency environments.

The shared headline manifests bind result hashes, code/model/runtime/workload
identity, hardware, environment and upstream rights. No model weights, tokenizer
payloads or inference runtime binaries are redistributed. No additional license is
asserted for model-derived output tokens.

These are contract qualification runs, not production performance or quality
benchmarks. The reports explicitly separate loaded-engine call latency from
startup and do not claim streaming TTFT, serving throughput, memory savings or
statistical significance. Review after 90 days and rerun after relevant engine,
model/tokenizer, measurement, dependency, driver or hardware changes.

Follow the [verification guide](../../../docs/guides/vllm-prefix-verification.md)
to prepare and run your own scoped comparison.
