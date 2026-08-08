# Metria

[![CI](https://github.com/dipeshbabu/metria/actions/workflows/ci.yml/badge.svg)](https://github.com/dipeshbabu/metria/actions/workflows/ci.yml)
[![Metria core](https://github.com/dipeshbabu/metria/actions/workflows/metria-core.yml/badge.svg)](https://github.com/dipeshbabu/metria/actions/workflows/metria-core.yml)
[![Root package](https://github.com/dipeshbabu/metria/actions/workflows/root-package.yml/badge.svg)](https://github.com/dipeshbabu/metria/actions/workflows/root-package.yml)

**Run, measure, and compare LLM inference systems with reproducible evidence.**

Metria is an open-source experiment and evidence layer for LLM inference
systems. It helps researchers answer a question that raw benchmark numbers do
not:

> For this model, runtime, workload, hardware, and treatment, what did we ask
> for, what actually ran, what changed, and are the resulting measurements
> valid to compare?

Metria sits **above** inference runtimes and optimization libraries. It is not a
serving engine, scheduler, kernel library, or universal quantizer. Systems such
as vLLM, llama.cpp, SGLang, MLX, TensorRT-LLM, torchao, LLM Compressor, and
custom research code remain responsible for execution; Metria provides the
study, provenance, measurement, and comparison layer around them.

> **Status:** Metria is under active development (`0.1.0.dev0`). The root
> package is installable from source, but it is not published to a package
> index yet. Public APIs should still be considered provisional.

## Why Metria

Inference research is easy to benchmark and surprisingly hard to compare well.
A result can look reproducible while hiding differences in model revisions,
tokenizers, runtime settings, cache formats, hardware, measurement methods, or
what the runtime actually applied.

Metria is being built around four principles:

1. **Requested is not observed.** Asking for FP8 KV cache is not proof that the
   runtime used FP8 KV cache.
2. **Comparability is study-specific.** A runtime or hardware change may be the
   variable under study rather than something that makes two runs globally
   incompatible.
3. **Measurements carry method identity.** Values with different methods or
   versions are not silently treated as the same metric.
4. **Failed and partial runs are evidence too.** Missing observation, timeout,
   unsupported configuration, or failed cleanup should not be converted into an
   apparently successful result.

## What works today

| Capability | Current state |
|---|---|
| Study design | `StudySpec`, `RunSpec`, `ComparisonPlan`, treatments, controls, blocking dimensions |
| Evidence model | Requested → resolved → observed state with immutable run evidence |
| Typed identity | `ModelRef`, `RuntimeConfig`, `WorkloadSpec`, `CapabilitySet`, `HardwareFingerprint`, `ArtifactManifest` |
| Capability inspection | Conservative model-geometry normalization, TurboQuant KV guardrails, and `metria inspect` |
| Hardware evidence | Privacy-conscious stdlib host/software fingerprinting; accelerator identity remains runtime-observed |
| Runtime lifecycle | `RuntimeAdapter` / `RuntimeSession` plus reusable runtime contract tests |
| First-party runtimes | llama.cpp and vLLM adapters |
| Execution | Failure-aware `execute_run()`, `execute_study()`, and durable `metria run` |
| Measurements | Decode-time token trajectory capture with retained prompt fingerprints |
| Pairwise analysis | KV Fidelity-compatible trajectory agreement analysis |
| Recipes | Versioned `metria.study_recipe.v1` JSON with deterministic SHA-256 digesting |
| Run records | Versioned `metria.run_record.v1` JSON with typed metrics plus full-record/evidence digests |
| Study results | `metria.study_result.v1` manifests with run digests, compatibility, analyses, recipe and hardware identity |
| Registries | Explicit built-ins via `metria plugins`; no arbitrary entry-point loading |
| CLI | Recipe tools, `inspect`, `compare`, `plugins`, and durable `run` |
| Packaging | Root `metria` package installable from source; focused components stay independent |

The standalone [KV Fidelity](components/kv-fidelity/README.md) package also
supports llama.cpp, MLX, vLLM, and SGLang for its focused KV-cache evaluation
workflow. Those component backends should not be confused with the smaller set
of first-party runtime adapters already exposed by the Metria core.

## Quick start

### 1. Install from source

```bash
git clone https://github.com/dipeshbabu/metria.git
cd metria
python -m pip install .

metria --version
metria --help
```

The root package deliberately does **not** install vLLM, llama.cpp, or other
inference engines. Runtime stacks remain optional and user-managed.

### 2. Define a study recipe

Metria recipes describe requested experiment intent as versioned data. For
example:

```json
{
  "schema": "metria.study_recipe.v1",
  "study": {
    "name": "runtime-comparison",
    "runs": [
      {
        "model": {"id": "example/model"},
        "runtime": {"name": "llamacpp"},
        "scenario": {"name": "decode"},
        "measurements": ["kv_fidelity.decode_time_trajectory"]
      },
      {
        "model": {"id": "example/model"},
        "runtime": {"name": "vllm"},
        "scenario": {"name": "decode"},
        "measurements": ["kv_fidelity.decode_time_trajectory"]
      }
    ],
    "comparison": {
      "vary": ["runtime"],
      "control": ["model", "scenario", "measurements"],
      "analyses": ["kv_fidelity.trajectory_match"]
    }
  },
  "measurement_configs": {
    "kv_fidelity.decode_time_trajectory": {
      "prompts": [
        {"id": "p1", "prompt": "The capital of France is"}
      ]
    }
  },
  "environment": {}
}
```

Validate, fingerprint, and inspect it:

```bash
metria recipe validate study.json
metria recipe digest study.json
metria inspect study.json
metria plugins
```

`metria plugins` lists only explicit built-ins. It does not scan arbitrary
Python entry points or install missing inference engines. vLLM can therefore be
reported unavailable when the optional package is absent, while llama.cpp is
recipe-dependent because the current adapter uses recipe-local binary/model
paths.

### 3. Execute into a durable evidence bundle

Use a new or empty output directory:

```bash
metria run study.json --output-dir results/experiment-a
```

A completed invocation writes:

```text
results/experiment-a/
  run-0000.json
  run-0001.json
  ...
  study-result.json
```

Each run file uses `metria.run_record.v1`. The manifest uses
`metria.study_result.v1` and retains recipe identity, hardware evidence, run
record/evidence digests, compatibility reports, and requested pairwise-analysis
outcomes.

Experimental failures are not silently dropped. If durable evidence is written
but the study contains failed/partial/timed-out runs, incompatible pairs, or
failed/skipped analyses, `metria run` exits `1`; invalid configuration or an
unavailable/unregistered implementation exits `2`.

See the [execution guide](docs/guides/metria-execution.md).

### 4. Compare saved run evidence

Compare persisted runs under the study's explicit `ComparisonPlan`:

```bash
metria compare run-0001.json run-0002.json --recipe study.json
metria compare run-0001.json run-0002.json --recipe study.json --json
```

Saved-record comparison deliberately requires the recipe. Record/evidence
digests identify serialized evidence; they do not replace study semantics.

See the [CLI guide](docs/guides/metria-recipe-cli.md) and
[run-record guide](docs/guides/metria-run-records.md).

## Core model

Metria does not model an experiment as a fixed list of six peer objects. The
study decides what varies and what must stay fixed:

```text
Study = factors + controls + comparison plan

Run = system under test
    × scenario / workload
    × measurement protocol
    @ observed environment
```

Every run then separates:

```text
requested
    ↓
resolved
    ↓
observed
    ↓
evidence + metrics + artifacts + lifecycle events
```

- **Requested** — what the user or recipe asked for.
- **Resolved** — exact settings, revisions, artifacts, and choices selected
  before launch.
- **Observed** — what the runtime and environment report actually ran.

That distinction is central to Metria. A configuration request is experiment
intent; observed state is evidence.

See [Metria core architecture](docs/architecture/metria-core.md) for the full
provisional contract.

## Comparison semantics

A `ComparisonPlan` declares the role of experiment dimensions:

```text
vary      dimensions intentionally changed
control   dimensions that must match
block_by  dimensions used to form comparable groups
```

Metria therefore does **not** use one universal "same fingerprint = comparable"
rule. Missing controlled evidence is not equality, and methodologically
different metrics require an explicit analysis that defines how they may be
combined.

## Runtime adapters

The Metria core currently includes two first-party runtime adapters:

### llama.cpp

The current adapter supports local llama.cpp command-line execution, GGUF model
paths, explicit runtime settings, KV-cache treatments, binary identity hashing,
requested/resolved/observed evidence, and optional decode-time token-ID capture
with a compatible patched binary.

See the [llama.cpp runtime guide](docs/guides/metria-llamacpp-runtime.md).

### vLLM

The current adapter uses the offline `vllm.LLM` API, keeps vLLM as an optional
lazy dependency, supports native token-ID capture, and separates configured
runtime state from introspected applied engine state.

See the [vLLM runtime guide](docs/guides/metria-vllm-runtime.md).

Runtime support is intentionally narrow while the common adapter contract is
being hardened. Metria should prefer adapters over reimplementing upstream
runtimes.

## Focused components

Metria remains a monorepo with focused components that keep their own package
identities and release lifecycles.

### KV Fidelity

Reference-anchored behavioral evaluation for KV-cache compression and runtime
changes.

```bash
python -m pip install "./components/kv-fidelity"
kv-fidelity --help
```

See [KV Fidelity](components/kv-fidelity/README.md) and its
[quick start](components/kv-fidelity/QUICKSTART.md).

### TurboQuant Reference

Portable NumPy/SciPy reference implementations of PolarQuant, QJL, and
TurboQuant KV-cache compression.

```bash
python -m pip install "./components/turboquant-reference"
python components/turboquant-reference/benchmarks/examples/demo.py
```

See [TurboQuant Reference](components/turboquant-reference/README.md).

## What Metria is not

Metria is not intended to become:

- another inference server;
- a replacement for vLLM, llama.cpp, SGLang, MLX, or TensorRT-LLM;
- a reimplementation of every quantization algorithm;
- a universal scalar fidelity score;
- an automatic recommendation engine before measurement uncertainty and
  failure semantics are mature;
- a single environment containing every inference runtime;
- a training-optimization framework.

## Roadmap

The current design roadmap is tracked in
[issue #50](https://github.com/dipeshbabu/metria/issues/50).

Near-term work is focused on:

1. **Runtime qualification** — exercise the shared contract against first-party
   adapters and add hardware-qualified evidence lanes where infrastructure is
   available.
2. **Observed runtime identity** — stronger served model/tokenizer/applied-config
   evidence.
3. **Artifact provenance** — immutable model/data verification and manifest
   identity.
4. **KV Fidelity migration** — route its comparison/reporting through shared
   Metria comparison semantics without breaking the focused package.
5. **Shared systems APIs** — move subprocess timeouts, benchmark orchestration,
   diagnostics, and reusable tooling behind Metria APIs rather than adding more
   standalone scripts.

Later phases can add broader evaluation suites, more runtime/optimization
adapters, experiment matrices, Pareto visualization, and constrained search.
Automatic recommendation should come only after the evidence layer is mature
enough to support it.

## Repository layout

```text
src/metria/                 Shared Metria core
metria_tests/               Core contract tests
components/
  kv-fidelity/              Focused fidelity evaluator
  turboquant-reference/     Portable algorithm reference
docs/                       Current guidance and architecture
research/                   Dated studies and investigations
artifacts/                  Retained experiment evidence
tools/                      Existing diagnostics and benchmark utilities
```

## Development

```bash
git clone https://github.com/dipeshbabu/metria.git
cd metria

uv sync --all-packages
uv run pre-commit install
uv run pytest
```

Core-only checks:

```bash
uv run pytest metria_tests -v
uv run mypy src/metria
uv run ruff check src/metria metria_tests
uv run ruff format --check src/metria metria_tests
```

Backend-specific inference dependencies remain optional. Install only the stack
needed for the runtime under test.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md), [GOVERNANCE.md](GOVERNANCE.md), and
[SUPPORT.md](SUPPORT.md).

New runtime, evaluator, benchmark, or optimization work should plug into the
common Metria contracts rather than creating a parallel architecture.

## Citation

Use [CITATION.cff](CITATION.cff) when Metria supports your work. When relying on
a specific result under `research/`, cite that report as well.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
