# Metria CLI

The `metria` command prepares, validates and runs reference/candidate
verification, and inspects or compares saved evidence. It uses only the standard
library; inference engines are installed separately.

| Command | Purpose | Guide |
| --- | --- | --- |
| `metria verify <study.json> --output <new-dir> [--json]` | Run a qualified reference/candidate change and write both records, `verification.json` and `report.md` | [Choose a workflow](verification-overview.md) |
| `metria demo --case {pass,fail,not-comparable} --output <dir>` | Write a synthetic bundle without a model | [Your first report](first-report.md) |
| `metria recipe prepare-vllm` | vLLM prefix caching off vs on; with `--kv-cache-dtype`, automatic vs FP8 KV cache (`--staged-kld` adds trajectory KL divergence) | [Prefix caching](vllm-prefix-verification.md), [FP8](fp8-kv-cache-verification.md) |
| `metria recipe prepare-vllm-upgrade` | Two pinned vLLM CPU environments | [Runtime upgrades](runtime-upgrade-verification.md) |
| `metria recipe prepare-vllm-serving` | Native streaming, throughput and allocation measurements | [Serving and memory](serving-measurements.md) |
| `metria recipe prepare-llamacpp-build` | Two pinned llama.cpp CPU capture builds | [CPU build changes](llamacpp-build-verification.md) |
| `metria recipe prepare-gguf-quantization` | A controlled GGUF Q8_0 conversion | [GGUF quantization](gguf-quantization-verification.md) |
| `metria recipe prepare-endpoint` | STAGED: two running OpenAI-compatible servers with a declared change | [Remote endpoints](endpoint-verification.md) |
| `metria recipe validate / digest / normalize <study.json>` | Check, fingerprint or canonicalize a recipe without running it | Below |
| `metria inspect <study.json> [--json]` | Evaluate capability rules and fingerprint local hardware | [Capability inspection](metria-inspection.md) |
| `metria compare <run.json> <run.json> ... --recipe <study.json>` | Compare saved run records under the recipe's plan | Below |
| `metria pilot record --evidence <dir> --notes <notes.json> --output <file>` | Bind a workload decision and feedback to a retained verification bundle | [Pilot checklist](verifier-pilot.md) |
| `metria fidelity {score,selftest,compare,repeatability,fetch}` | Expert KV-cache fidelity scoring and saved-report tools | [Fidelity commands](unified-fidelity.md) |

`metria --version` prints the installed version, and `python -m metria` is an
equivalent entry point. Exit codes for `verify` are listed in
[policies](verification-policies.md#evaluation-order-and-ci-exits).

## Install

```bash
python -m pip install metria
metria --version
```

From a source checkout, `python -m pip install .` installs the same command.
With the uv workspace:

```bash
uv sync --all-packages
uv run metria --version
```

## Validate

```bash
metria recipe validate study.json
```

Successful human-readable output includes:

- recipe schema;
- study name;
- canonical recipe digest.

It does not print prompt text.

For structural machine-readable metadata:

```bash
metria recipe validate study.json --json
```

The JSON output contains the study name, run count, runtime names, measurement
names, analysis names, schema, and digest. It intentionally does not reproduce
measurement configs or prompt contents.

Validation failures return a non-zero exit status with a concise error on
stderr rather than a Python traceback.

## Digest

```bash
metria recipe digest study.json
```

This prints only the canonical SHA-256 digest from `study_recipe_digest()`.
The digest identifies requested serialized intent; it is not proof that a
runtime applied the request.

## Normalize

```bash
metria recipe normalize study.json
```

or:

```bash
metria recipe normalize study.json --output normalized-study.json
```

`normalize` validates the recipe and emits deterministic human-readable JSON.

### Privacy warning

Unlike `validate` and `digest`, **normalize reproduces the complete recipe**.
If measurement configs contain raw prompts, system messages, private paths, or
other sensitive inputs, normalized output contains them too.

Do not pipe normalized private recipes into public logs or publish them under
`artifacts/` unless those inputs are intentionally public.

## Inspect

```bash
metria inspect study.json
metria inspect study.json --json
```

Inspection evaluates data-only capability rules before execution and captures a
privacy-conscious local hardware/software fingerprint. It does not infer model
geometry from a model name and does not treat an environment variable as proof
that an accelerator exists.

See [capability inspection](metria-inspection.md).

## Compare saved records

```bash
metria compare run-0001.json run-0002.json --recipe study.json
```

Use `--json` for machine-readable pairwise output. The recipe is mandatory
because comparability is study-specific; the command does not invent a global
fingerprint rule.

The records must belong to the recipe supplying the `ComparisonPlan`. A valid
but incompatible comparison returns exit status `1`; malformed input or a
record/recipe binding error returns `2`.

See [run records and comparison](metria-run-records.md).

## Python API

Studies beyond the qualified `verify` profiles run through `execute_run()` and
`execute_study()`; see [run records](metria-run-records.md) and
[pairwise analyses](metria-pairwise-analysis.md). Installing the CLI does not
install an inference engine or auto-load third-party plugins.
