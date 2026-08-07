# Metria

Metria is an open-source evidence and experimentation layer for LLM inference
systems.

It is designed to help researchers define inference-system studies, verify what
actually ran, measure systems and behavioral effects under explicit protocols,
and determine when results can or cannot be validly compared.

Metria is **not** another serving engine or universal quantization library.
Production runtimes and optimization implementations should normally stay in
projects such as vLLM, SGLang, llama.cpp, MLX, TensorRT-LLM, torchao, LLM
Compressor, or custom research code. Metria provides the experiment and evidence
layer around them.

## Direction

The core model is deliberately study-oriented:

```text
Study = factors + controls + comparison plan

Run = system under test
    × scenario
    × measurement protocol
    @ observed environment
```

A run record keeps three states separate:

1. **Requested** — what the recipe asked for.
2. **Resolved** — the exact configuration and artifacts selected before launch.
3. **Observed** — what the runtime and environment report as having actually run.

This distinction matters for inference research: requesting a cache dtype,
model revision, attention backend, or kernel path is not evidence that the
launched system actually used it.

See [Metria core architecture](docs/architecture/metria-core.md) for the
provisional study, run, metric, treatment, and comparison contracts.

## First milestone

The initial stable target is intentionally narrow:

> Metria can reproducibly run, record, and validly compare a defined
> inference-systems study across at least two runtimes.

Automated recommendation, active search, a universal fidelity score, production
serving, and a universal all-runtime dependency environment are explicitly
outside the first milestone.

## Current repository

Metria is a research monorepo with components at different maturity levels.

| Area | Purpose | Status |
|---|---|---|
| `src/metria/` | Shared study, run, metric, comparison, runtime, and measurement contracts | Provisional foundation |
| [KV Fidelity](components/kv-fidelity/README.md) | Reference-anchored KV-cache behavioral evaluation across llama.cpp, MLX, vLLM, and SGLang | Beta; independent focused distribution |
| [TurboQuant Reference](components/turboquant-reference/README.md) | NumPy/SciPy reference implementation of PolarQuant, QJL, and TurboQuant KV-cache compression | Research reference; independent focused distribution |
| [Tools](tools/README.md) | Existing diagnostics, validation, benchmarking, and conversion utilities | Mixed; being migrated into reusable protocols over time |
| [Research](research/README.md) | Dated studies, investigations, negative results, and archived plans | Evidence record |
| [Artifacts](artifacts/README.md) | Retained benchmark output and experiment evidence | Evidence; provenance varies for historical artifacts |

KV Fidelity and TurboQuant Reference keep their own package names and release
lifecycles while the shared Metria layer matures.

## What Metria should own

Metria's differentiated responsibilities are:

- explicit study design: what varies, what is controlled, and what is blocked;
- requested, resolved, and observed configuration provenance;
- runtime lifecycle and applied-configuration evidence;
- measurement protocols with typed units, methods, samples, aggregation, and
  uncertainty;
- comparison semantics that prevent invalid cross-system conclusions;
- behavioral and systems metrics under one evidence model;
- hardware-aware capability discovery;
- reproducibility bundles for research results.

Metria should generally **not** own inference kernels, schedulers, serving
engines, or duplicate every upstream quantizer.

## Development setup

Clone the repository and synchronize the workspace:

```bash
git clone https://github.com/dipeshbabu/metria.git
cd metria

uv sync --all-packages
uv run pre-commit install
uv run pytest
```

Run the provisional Metria core checks directly with:

```bash
uv run pytest metria_tests -v
uv run mypy src/metria
uv run ruff check src/metria metria_tests
uv run ruff format --check src/metria metria_tests
```

Backend-specific KV Fidelity dependencies remain optional. Add only the stack
needed for the backend under test; do not assume all inference runtimes can
share one Python environment.

## Existing focused workflows

### Evaluate KV-cache fidelity

Install the focused component from a checkout:

```bash
python -m pip install "./components/kv-fidelity"
kv-fidelity --help
```

Continue with the [KV Fidelity quick start](components/kv-fidelity/QUICKSTART.md).

### Reproduce TurboQuant algorithms

```bash
python -m pip install "./components/turboquant-reference"
python components/turboquant-reference/benchmarks/examples/demo.py
```

See the [TurboQuant Reference README](components/turboquant-reference/README.md)
for its public Python API and benchmark extras.

### Inspect existing diagnostic tools

```bash
python tools/diagnostics/turbo_hardware_diag.py --help
```

The current scripts contain useful operational knowledge but are not yet the
stable Metria benchmark protocol. See [tools/README.md](tools/README.md) for
requirements and limitations.

## Repository organization

```text
src/metria/                 Provisional shared Metria core
metria_tests/               Core contract tests
components/
  kv-fidelity/              Focused fidelity evaluator
  turboquant-reference/     Portable algorithm reference
docs/                       Current guidance and architecture
research/                   Dated research and investigations
artifacts/                  Retained experiment evidence
tools/                      Existing diagnostics and benchmark utilities
```

Current guidance belongs in `docs/`. Dated claims and negative results belong
in `research/`. Generated evidence belongs in `artifacts/`. Historical evidence
is preserved rather than rewritten to match newer conclusions.

## Verification

Run the complete repository gate:

```bash
uv run pre-commit run --all-files
uv run pytest
```

Focused components can still be built independently:

```bash
uv run python -m build components/kv-fidelity
uv run python -m build components/turboquant-reference
```

The root Metria core is still provisional and is not published from this branch.
Public contracts should remain explicitly unstable until the runtime and
measurement boundaries have been exercised by at least two materially
different inference runtimes.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md), [GOVERNANCE.md](GOVERNANCE.md), and
[SUPPORT.md](SUPPORT.md). Changes to public experiment semantics, metric
methodology, comparison rules, packaging, or release policy should be discussed
before being stabilized.

## Citing this work

Use [CITATION.cff](CITATION.cff) when the repository as a whole supports your
work. When relying on a specific result under `research/`, cite that report as
well so readers can recover its model, runtime, hardware, configuration, and
date.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
