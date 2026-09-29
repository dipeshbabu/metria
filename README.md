# Metria

[![PyPI](https://img.shields.io/pypi/v/metria)](https://pypi.org/project/metria/)
[![CI](https://github.com/dipeshbabu/metria/actions/workflows/ci.yml/badge.svg)](https://github.com/dipeshbabu/metria/actions/workflows/ci.yml)
[![Metria core](https://github.com/dipeshbabu/metria/actions/workflows/metria-core.yml/badge.svg)](https://github.com/dipeshbabu/metria/actions/workflows/metria-core.yml)

**Metria verifies changes to LLM inference systems.**

Give it a reference configuration and a candidate configuration. Metria checks
what actually ran, rejects an unjustified comparison, measures behavioral change,
and evaluates the result against your explicit acceptance criteria.

**Test inference changes before you ship them.**

The current end-to-end CLI verifies a local llama.cpp CPU thread-count change.
It holds the pinned model, qualified capture provider, prompt workload, and other
settings fixed, checks native readback, and retains both runs with a reviewable
report. The Python runtime adapters and research components support additional
building blocks; their presence does not establish a qualified CLI workflow.

## Start with one change

For the current development features, install a checkout with Python 3.10–3.14:

```bash
git clone https://github.com/dipeshbabu/metria.git
cd metria
python -m pip install .
```

Prepare the pinned runtime, model, and reference/candidate recipe using the
[local verification guide](docs/guides/metria-verify.md), then run:

```bash
metria verify study.json --output verification --json
```

The root Python package has no runtime dependencies. Native engines and models
are installed separately. Each output directory must be new.

Published [Metria 0.1.1](https://github.com/dipeshbabu/metria/releases/tag/metria-v0.1.1)
includes canonical results, diagnostics, explicit policies, compatible performance
deltas, and CI integration. The 0.2 development checkout also unifies fidelity
methods under `metria.fidelity` and `metria fidelity`; see the
[migration guide](docs/guides/unified-fidelity.md). It also provides
[qualified vLLM prefix-cache verification](docs/guides/vllm-prefix-verification.md)
with retained CPU/CUDA evidence, cache isolation and loaded-engine latency.

In the 0.2 development checkout, try the report flow without a model:

```bash
uv run metria demo --case pass --output demo-pass
uv run metria recipe prepare-vllm --help
```

The demo is explicitly synthetic. Use the [installed workflow](docs/guides/installed-workflow.md)
for your own pinned runtime/model and workload, then add
[task checks and an acceptance policy](docs/guides/verification-decisions.md).

Start with the [copyable workflows](examples/verification/README.md): a qualified
local configuration change, synthetic PASS/FAIL/invalid-comparison fixtures,
and clearly staged templates for runtime upgrades, KV precision, quantization,
and build regressions. Synthetic examples require no model or GPU and are
labeled as test evidence throughout.

## Read the decision

Every result separates four questions:

| Section | What it answers |
|---|---|
| Change | What intentionally differs between reference and candidate? |
| Evidence | What ran, what stayed controlled, and what evidence is missing? |
| Impact | Where did token trajectories diverge, and what compatible systems impact was observed? |
| Verdict | Did execution finish, was comparison valid, and did your policy pass? |

```text
verification/
  verification.json
  report.md
  reference.run.json
  candidate.run.json
```

The original `manifest.json` filename remains a compatibility alias. The canonical
JSON and Markdown report include separate lifecycle, comparison, and policy
states. Timeouts, failed runs, insufficient evidence, and unexpected differences
remain visible instead of becoming a successful comparison.

Metria distinguishes **requested → resolved → observed**: a requested setting
states intent; the resolved artifact/configuration identifies what will launch;
native observation checks what actually applied. A requested CPU thread change
must appear in runtime readback before its behavioral result is accepted. An
undeclared tokenizer or generation change prevents comparison.

The first behavioral method is KV Fidelity-compatible token-trajectory analysis.
It reports prefix agreement, exact matches, first-divergence positions, category
rates, and prompt identifiers that help locate drift without dumping prompt text.
See [behavioral diagnostics](docs/guides/metria-trajectory-measurement.md).

Systems measurements have distinct boundaries: llama.cpp cold-process latency,
loaded-engine vLLM request latency, and
[local serving measurements](docs/guides/serving-measurements.md) for streaming,
concurrency and native CUDA allocator/KV storage. Reports retain method, workload,
coverage and limitations; missing native measurements remain unavailable.
See [performance evidence](docs/guides/verification-performance.md).

`VERIFIED` means valid comparison and completed analysis. Add an
[explicit acceptance policy](docs/guides/verification-policies.md) for PASS/FAIL.
Metria supplies no universal behavior or safety threshold; PASS means your stated
criteria were met after the evidence gates passed.

## Use it in code review and CI

Run the same command as a job step and preserve its exit status:

```yaml
- name: Verify inference change
  run: metria verify .metria/change.json --output metria-verification --json
```

The [CI guide](docs/guides/verification-ci.md) defines distinct failure exits,
retained artifacts, and a GitHub Actions summary adapter. The
[copyable workflow](examples/verification/github-actions.yml) keeps the report
and records even when verification fails. A real CI workload needs the same
qualified runtime/model provisioning as local use.

## Scope and qualification

| Surface | Scope |
|---|---|
| `metria verify` | Qualified local llama.cpp CPU thread changes, plain greedy completion, pinned model/provider, one reference and one candidate |
| Runtime adapters | llama.cpp and vLLM Python APIs; qualification applies only to retained runtime/model/device configurations |
| Supporting commands | Recipe validation/digests, capability inspection, and saved-record comparison for preparation and debugging |
| KV Fidelity | Independently versioned behavioral methodology/component consumed by Metria's verification path |
| TurboQuant Reference | Portable research/reference implementation with its own lifecycle |

See [runtime qualification](docs/guides/runtime-qualification.md). A mocked
contract test, installed engine, or visible GPU is not a real-engine qualification.
New routes must prove their identity, requested change, and required captures.

Metria focuses on one inference change at a time. Broad experiment matrices,
generic benchmark orchestration, arbitrary plugin discovery, automatic search,
Pareto dashboards, additional runtimes solely for breadth, training optimization,
and hosted services are deferred until they directly strengthen this verifier.
The product decision is tracked in [#50](https://github.com/dipeshbabu/metria/issues/50).

## Supporting APIs and repository

Use the [recipe CLI](docs/guides/metria-recipe-cli.md),
[inspection guide](docs/guides/metria-inspection.md), and
[saved-record comparison](docs/guides/metria-run-records.md) to prepare or debug
a verification. The [core architecture](docs/architecture/metria-core.md) documents
the reusable study/run/evidence machinery behind that workflow.

`src/metria/` owns verification and shared evidence contracts.
`components/kv-fidelity/` owns focused behavioral methods;
`components/turboquant-reference/` owns the research reference implementation.
Current guidance lives in `docs/`; dated investigations in `research/`; retained
evidence in `artifacts/`. Historical results keep their original scope and known
gaps under the [provenance policy](docs/maintainers/third-party-material.md).

For development:

```bash
uv sync --locked --all-packages
uv run pre-commit install
uv run python -m pytest
```

See [Contributing](CONTRIBUTING.md), [Governance](GOVERNANCE.md), and
[Support](SUPPORT.md). Cite [CITATION.cff](CITATION.cff) and the specific retained
report when relying on an experimental result.

Original software uses Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
Third-party/model-derived material retains its separately documented rights.
