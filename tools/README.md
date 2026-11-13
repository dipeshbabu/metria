# Repository tools

Use the installed `metria` CLI for inference-change verification. Start with
[the installed workflow](../docs/guides/installed-workflow.md) or the
[workflow chooser](../docs/guides/verification-overview.md).

This directory contains source-checkout utilities for qualification, repeated
trials, diagnostics, research validation, documentation, and repository checks.

## Qualification

The [qualification guide](qualification/README.md) documents pinned native builds,
capture providers, model descriptors, and retained evidence.

| Utility | Purpose |
| --- | --- |
| `qualification/build_llamacpp_cpu.sh` | Build the qualified native CPU capture provider. |
| `qualification/prepare_cpu_verification.py` | Prepare a pinned CPU thread comparison. |
| `qualification/prepare_llamacpp_build_verification.py` | Prepare a comparison between independently pinned CPU builds. |
| `qualification/prepare_gguf_quantization.py` | Prepare controlled GGUF weight conversion and verification. |
| `qualification/prepare_vllm_prefix_verification.py` | Prepare the qualified prefix-cache comparison. |
| `qualification/prepare_vllm_upgrade.py` | Prepare isolated CPU runtime-upgrade environments. |
| `qualification/qualify_vllm.py` | Retain native vLLM qualification evidence. |
| `qualification/staged/` | Observe experimental treatments outside qualified end-to-end workflows. |

Preparation helpers do not download models or install inference engines. Read the
owning workflow's qualification requirements before using a helper.

## Repeated verification trials

Run the same bounded verifier lifecycle for each pair:

```bash
python tools/benchmarks/verification_trials.py \
  --recipe study.json --output trials \
  --warmup-pairs 1 --measured-pairs 3
```

The `turbo-quick-bench.sh` and `turbo-realworld-bench.sh` entry points forward to
this utility. See [trial policy and migration](../docs/guides/verification-trials.md)
for native timing, exact baseline identity, and retired legacy arguments.

## Optional research diagnostics

Inspect the platform contract before starting a hardware diagnostic:

```bash
python tools/diagnostics/turbo_hardware_diag.py --platform-support
```

| Host | Diagnostic support |
| --- | --- |
| Apple Silicon macOS | Metal. |
| Native Linux | CUDA; ROCm, Vulkan, and CPU have experimental boundaries. |
| WSL | CUDA with reduced host telemetry; use Linux runtime binaries. |
| Native Windows or Intel macOS | Unsupported by this diagnostic. |

These are diagnostic boundaries, not the Metria core's platform contract.
Historical TurboQuant measurements require their original experimental engine
forks. The [fork registry](../docs/reference/historical-forks.md) records missing
sources and reproducibility limits; current Metria verification does not require
those historical forks.

The `diagnostics/turbo-diag` launcher can install optional Rich UI support in a
local environment. Running the Python diagnostic directly does not install
dependencies. Both use Metria's [bounded subprocess lifecycle](../docs/guides/subprocess-lifecycle.md).

Diagnostic bundles contain a text log, structured hardware profile, and monitoring
CSV. Review paths, host details, and native output before sharing. Retain source
identity and describe any redaction when a bundle supports a reported result.

Research validators live in `validation/`. Benchmark results and failed
experiments belong in the [research record](../research/README.md) and
[artifact collection](../artifacts/README.md), rather than in this tools index.

## Documentation

```bash
uv run --locked --script tools/docs/site.py build --strict
python tools/docs/check_site.py site
```

See [documentation maintenance](../docs/maintainers/documentation.md) for the local
preview, locked dependencies, style rules, and GitHub Pages deployment.

## Maintenance and CI

`maintenance/` contains the checks used by repository hooks and CI: local links,
portable paths, complexity and coverage floors, citation metadata, strict JSON,
package boundaries, release artifacts, provenance, and issue/PR labels.

Run the configured checks from the repository root:

```bash
uv run --locked pre-commit run --all-files
```

`ci/` contains Python 3.10 compatibility tooling and verification summaries for
GitHub Actions. Package publication follows the separate
[release procedure](../docs/guides/releasing-metria.md).
