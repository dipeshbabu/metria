<h1 align="center">Metria</h1>

<p align="center">
  Test inference changes before you ship them.
</p>

<p align="center">
  <a href="https://pypi.org/project/metria/"><img alt="PyPI" src="https://img.shields.io/pypi/v/metria?style=flat-square&amp;color=3ECF8E"></a>
  <a href="https://github.com/dipeshbabu/metria/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/dipeshbabu/metria/ci.yml?branch=main&amp;style=flat-square&amp;label=CI&amp;color=3ECF8E"></a>
  <a href="https://github.com/dipeshbabu/metria/actions/workflows/metria-core.yml"><img alt="Metria core" src="https://img.shields.io/github/actions/workflow/status/dipeshbabu/metria/metria-core.yml?branch=main&amp;style=flat-square&amp;label=core&amp;color=3ECF8E"></a>
  <a href="https://github.com/dipeshbabu/metria/blob/main/LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-3ECF8E?style=flat-square"></a>
</p>

---

Metria verifies changes to LLM inference systems. Give it a reference and a
candidate configuration to check what actually ran, measure behavioral change,
and evaluate the result against your acceptance criteria. Keep both runs and a
report you can review locally or in CI.

## Installation

Use Python 3.10–3.14 in a virtual environment:

```sh
python -m pip install metria==0.1.5
metria --version
```

Metria 0.1.5 is an alpha release. The base package has no runtime dependencies;
install native inference engines and models separately for real verification.
See the [release notes](https://github.com/dipeshbabu/metria/blob/metria-v0.1.5/docs/releases/0.1.5.md).

## Try the Report

Run a complete synthetic example without a model or GPU:

```sh
metria demo --case pass --output demo-pass
```

Open `demo-pass/report.md` to see the decision, evidence checks, and behavioral
diagnostics. Try `--case fail` or `--case not-comparable` with a new output
directory to see a policy failure or an invalid comparison. These examples are
labeled synthetic throughout; they do not establish runtime support, model
quality, or performance.

## Verify an Inference Change

Start with the [installed workflow](https://github.com/dipeshbabu/metria/blob/metria-v0.1.5/docs/guides/installed-workflow.md)
to prepare a qualified runtime, a pinned local model, and a workload. For a vLLM
prefix-cache comparison, run from your prepared runtime environment:

```sh
metria recipe prepare-vllm \
  --model /path/to/local/model \
  --descriptor trusted-model-files.json \
  --workload prompts.jsonl \
  --policy policy.json \
  --output study.json

metria recipe validate study.json
metria verify study.json --output verification --json
```

The descriptor pins your model and tokenizer files; the policy defines your
acceptance criteria. Preparation checks those inputs and the installed runtime
before writing the recipe. It does not download models or install engines.
Choose a new recipe filename and a new evidence directory for each comparison.

Metria checks **requested → resolved → observed** configuration: the setting you
asked for, the artifacts and configuration selected to launch, and native evidence
of what actually applied. Missing readback or an undeclared tokenizer or
generation change prevents a successful comparison.

These profiles have retained native qualification within their documented scope:

| Change | Guide |
| --- | --- |
| llama.cpp CPU thread count | [Local configuration](https://github.com/dipeshbabu/metria/blob/metria-v0.1.5/docs/guides/metria-verify.md) |
| llama.cpp CPU build | [Build regression](https://github.com/dipeshbabu/metria/blob/metria-v0.1.5/docs/guides/llamacpp-build-verification.md) |
| GGUF Q8_0 weight quantization | [Quantization](https://github.com/dipeshbabu/metria/blob/metria-v0.1.5/docs/guides/gguf-quantization-verification.md) |
| vLLM prefix caching | [Prefix-cache verification](https://github.com/dipeshbabu/metria/blob/metria-v0.1.5/docs/guides/vllm-prefix-verification.md) |
| vLLM CPU runtime version | [Runtime upgrade](https://github.com/dipeshbabu/metria/blob/metria-v0.1.5/docs/guides/runtime-upgrade-verification.md) |
| vLLM FP8 KV cache (CUDA, SM89+) | [FP8 KV-cache verification](https://github.com/dipeshbabu/metria/blob/main/docs/guides/fp8-kv-cache-verification.md) |
| vLLM local serving concurrency | [Serving measurements](https://github.com/dipeshbabu/metria/blob/metria-v0.1.5/docs/guides/serving-measurements.md) |

Each profile has specific runtime, model, device, and evidence requirements.
See [runtime qualification](https://github.com/dipeshbabu/metria/blob/metria-v0.1.5/docs/guides/runtime-qualification.md)
before using a different configuration.

## Read the Decision

A completed bundle contains the canonical result, a readable report, and both run
records:

```text
verification/
  verification.json
  report.md
  reference.run.json
  candidate.run.json
```

The report describes the intended change, checks execution evidence and comparison
controls, then reports behavioral and compatible systems impact. It keeps
execution status, comparison validity, and policy acceptance separate.
`manifest.json` remains a compatibility alias for `verification.json`.

| Outcome | Meaning | Exit |
| --- | --- | --- |
| `VERIFIED` | Execution and valid behavioral comparison completed; no acceptance policy was evaluated. | `0` |
| `PASS` / `FAIL` | A valid comparison met / did not meet your explicit acceptance policy. | `0` / `1` |
| Invalid input | The recipe or configuration could not be used. | `2` |
| `NOT_COMPARABLE` | An undeclared or controlled difference prevents comparison. | `3` |
| `INSUFFICIENT_EVIDENCE` | Required identity, runtime readback, or captures are missing or inconsistent. | `4` |
| `EXECUTION_FAILED` | Execution, preflight, analysis, or persistence failed. | `5` |
| Interrupted | Verification was interrupted. | `130` |

`VERIFIED` establishes that comparison completed. For a keep/reject decision, add
[task checks and an acceptance policy](https://github.com/dipeshbabu/metria/blob/metria-v0.1.5/docs/guides/verification-decisions.md).
PASS applies to your stated criteria on the supplied workload; Metria supplies no
universal behavior or safety threshold.

[Behavioral diagnostics](https://github.com/dipeshbabu/metria/blob/metria-v0.1.5/docs/guides/metria-trajectory-measurement.md)
include token-prefix agreement, exact matches, and first-divergence positions.
[Systems measurements](https://github.com/dipeshbabu/metria/blob/metria-v0.1.5/docs/guides/verification-performance.md)
retain their method and limitations: cold-process llama.cpp latency,
loaded-engine vLLM latency, or local serving measurements. Missing native
measurements remain unavailable.

## Use in CI

Provision the qualified runtime and model, then run the same verification command
as a job step:

```yaml
- name: Verify inference change
  run: metria verify .metria/change.json --output metria-verification --json
```

Preserve the exit status and retain the evidence directory even when verification
fails. The [copyable GitHub Actions workflow](https://github.com/dipeshbabu/metria/blob/metria-v0.1.5/examples/verification/github-actions.yml)
includes report summaries and artifact retention. See the
[CI guide](https://github.com/dipeshbabu/metria/blob/metria-v0.1.5/docs/guides/verification-ci.md)
for failure handling and machine-readable results.

## Reference

Use `--help` to explore commands and options:

```sh
metria verify --help
metria recipe --help
metria inspect --help
metria fidelity --help
```

- [Copyable verification examples](https://github.com/dipeshbabu/metria/blob/metria-v0.1.5/examples/verification/README.md)
- [Recipe CLI](https://github.com/dipeshbabu/metria/blob/metria-v0.1.5/docs/guides/metria-recipe-cli.md)
- [Capability inspection](https://github.com/dipeshbabu/metria/blob/metria-v0.1.5/docs/guides/metria-inspection.md)
- [Saved-record comparison](https://github.com/dipeshbabu/metria/blob/metria-v0.1.5/docs/guides/metria-run-records.md)
- [Fidelity methods and migration](https://github.com/dipeshbabu/metria/blob/metria-v0.1.5/docs/guides/unified-fidelity.md)
- [Verifier pilot checklist](https://github.com/dipeshbabu/metria/blob/metria-v0.1.5/docs/guides/verifier-pilot.md)

## Developing

Use Python 3.11–3.14 and uv 0.11.8 or newer for the development workspace. From a
fresh checkout:

```sh
uv sync --locked --all-packages
uv run pre-commit install
uv run --locked pre-commit run --all-files
uv run --locked python -m pytest
```

Useful source entry points:

| Path | Purpose |
| --- | --- |
| `src/metria/` | CLI, verification, and shared evidence contracts |
| `src/metria/fidelity/` | Integrated behavioral methods and assets |
| `components/kv-fidelity/` | Method documentation, characterization tests, and source compatibility bridge |
| `components/turboquant-reference/` | Portable research reference implementation |
| `docs/` | Current guides and architecture |
| `tools/` | Qualification, diagnostics, and maintenance utilities |
| `artifacts/` | Retained qualification and experiment evidence |

Runtime dependencies stay optional. Install only the backend stack you need;
native backend extras require Python 3.11 or newer to use patched dependencies.
The dependency-free core retains Python 3.10 support, tested with a separate
locked compatibility environment.
See [Contributing](https://github.com/dipeshbabu/metria/blob/main/CONTRIBUTING.md)
for platform setup, coverage gates, and distribution checks.

## Contributing

Focus contributions on one inference change or evidence failure. Open a design
issue before changing public contracts, methodology, dependencies, or release
policy. See [Contributing](https://github.com/dipeshbabu/metria/blob/main/CONTRIBUTING.md),
[Governance](https://github.com/dipeshbabu/metria/blob/main/GOVERNANCE.md), and
[Support](https://github.com/dipeshbabu/metria/blob/main/SUPPORT.md).

When citing an experimental result, use
[CITATION.cff](https://github.com/dipeshbabu/metria/blob/main/CITATION.cff) and the
specific retained report. Historical evidence keeps its original scope and known
gaps under the [provenance policy](https://github.com/dipeshbabu/metria/blob/main/docs/maintainers/third-party-material.md).

## License

Original software is released under [Apache-2.0](https://github.com/dipeshbabu/metria/blob/main/LICENSE).
See [NOTICE](https://github.com/dipeshbabu/metria/blob/main/NOTICE) for attribution.
Third-party and model-derived material retains its separately documented rights.
