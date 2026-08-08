# Metria CLI

Metria's current command line is deliberately data/evidence oriented. It can
validate versioned study recipes, inspect requested capabilities/hardware, and
compare already-saved run records. It does not yet execute studies, auto-load
plugins, or make optimization recommendations.

Current commands include:

```text
metria recipe validate <study.json>
metria recipe digest <study.json>
metria recipe normalize <study.json>
metria inspect <study.json>
metria compare <run.json> <run.json> [...] --recipe <study.json>
```

The implementation uses only the standard library and the versioned Metria JSON
schemas.

## Install from a source checkout

The root Metria framework is installable from a checkout and provides a real
`metria` console command:

```bash
git clone https://github.com/dipeshbabu/metria.git
cd metria
python -m pip install .
metria --version
metria recipe validate study.json
```

The root distribution remains provisional and is not published by this
repository. The local package metadata is not a claim that a public package
index namespace is owned or available.

Developers using the uv workspace can instead run:

```bash
uv sync --all-packages
uv run metria --version
uv run metria recipe validate study.json
```

`python -m metria` remains an equivalent module entry point after installation.

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

## Why execution is not here yet

Study execution requires explicit runtime, measurement, and pairwise-analysis
registries plus durable output/provenance policy. Installing the root CLI does
not auto-import arbitrary plugins or install inference engines.

The remaining sequence is:

1. keep recipe, inspection, and run-record schemas versioned and strict;
2. add explicit built-in registry selection/listing;
3. attach recipe/hardware identities to execution provenance;
4. add `metria run` with durable `metria.run_record.v1` output;
5. keep third-party plugin loading separate from the first execution CLI.
