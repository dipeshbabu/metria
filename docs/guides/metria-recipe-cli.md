# Metria CLI

Metria's command line is evidence-oriented. It can validate versioned study
recipes, inspect requested capabilities/hardware, list explicit built-ins,
execute studies into durable evidence bundles, and compare saved run records.
It does not auto-load arbitrary third-party entry points, auto-install inference
engines, or make optimization recommendations.

Current commands include:

```text
metria recipe validate <study.json>
metria recipe digest <study.json>
metria recipe normalize <study.json>
metria inspect <study.json>
metria plugins
metria run <study.json> --output-dir <directory>
metria compare <run.json> <run.json> [...] --recipe <study.json>
```

## Install from a source checkout

```bash
git clone https://github.com/dipeshbabu/metria.git
cd metria
python -m pip install .
metria --version
```

The root distribution remains provisional and is not published by this
repository. Runtime dependencies remain optional and user-managed.

## Recipe validation and normalization

```bash
metria recipe validate study.json
metria recipe validate study.json --json
metria recipe digest study.json
metria recipe normalize study.json --output normalized-study.json
```

`validate` and `digest` do not reproduce prompt text. `normalize` deliberately
reproduces the complete validated recipe, which can contain prompts, private
paths, or other sensitive input.

## Inspect

```bash
metria inspect study.json
metria inspect study.json --json
```

Inspection evaluates data-only capability rules and captures a
privacy-conscious local hardware/software fingerprint. It does not infer model
geometry from a model name and does not treat an environment variable as proof
that an accelerator exists.

See [capability inspection](metria-inspection.md).

## List built-ins

```bash
metria plugins
metria plugins --json
```

The first registry is explicit. It lists built-in runtimes, measurements, and
pairwise analyses together with static availability:

- `available` — the registry can use the implementation in the current Python
  environment;
- `recipe_dependent` — availability depends on recipe-local paths/settings;
- `unavailable` — a required optional dependency is absent.

The command does not import arbitrary Python entry points or install missing
runtime packages.

See [execution and registries](metria-execution.md).

## Execute a study

```bash
metria run study.json --output-dir results/experiment-a
metria run study.json --output-dir results/experiment-a --json
```

The output directory must be new or empty. A completed invocation writes one
`metria.run_record.v1` file per run and a `metria.study_result.v1` manifest.

The manifest records recipe identity, hardware evidence, run statuses,
record/evidence digests, compatibility reports, and requested pairwise-analysis
outcomes. Experimental failures remain durable evidence rather than being
silently dropped.

Execution exit status is:

- `0` — all runs completed, all generated pairs are compatible, and requested
  analyses completed;
- `1` — durable output exists but the study includes failed/partial/timed-out
  runs, incompatible pairs, or failed/skipped analyses;
- `2` — invalid configuration, unavailable/unregistered implementation,
  non-empty output directory, or another command error before a durable result
  is produced.

See [execution and registries](metria-execution.md).

## Compare saved records

```bash
metria compare run-0001.json run-0002.json --recipe study.json
metria compare run-0001.json run-0002.json --recipe study.json --json
```

The recipe is mandatory because comparability is study-specific. Records must
bind to the study and requested `RunSpec` that supply the `ComparisonPlan`.

A valid but incompatible comparison returns `1`; malformed input or a
record/recipe binding error returns `2`.

See [run records and comparison](metria-run-records.md).

## Execution trust boundary

Recipes name implementations but do not contain Python import paths. The first
execution CLI resolves names only through Metria's built-in registry. A future
third-party plugin mechanism should define a separate trust/loading policy
rather than silently extending this command.
