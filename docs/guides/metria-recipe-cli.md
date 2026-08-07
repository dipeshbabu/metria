# Metria recipe CLI

Metria's first command-line surface is deliberately limited to versioned recipe
inspection. It does not execute studies, auto-load plugins, or make optimization
recommendations.

The initial commands are:

```text
metria recipe validate <study.json>
metria recipe digest <study.json>
metria recipe normalize <study.json>
```

The implementation uses only the standard library and the existing
`metria.study_recipe.v1` schema.

## Running from a source checkout

The root Metria distribution is still provisional and is not yet installed as a
console-script package by this change. From a repository checkout, run:

```bash
PYTHONPATH=src python -m metria --version
PYTHONPATH=src python -m metria recipe validate study.json
```

On Windows PowerShell, the equivalent is:

```powershell
$env:PYTHONPATH = "src"
python -m metria recipe validate study.json
```

Packaging the root Metria core and installing a `metria` console script should
be a separate change so packaging/release semantics are reviewed independently
from the CLI contract.

## Validate

```bash
PYTHONPATH=src python -m metria recipe validate study.json
```

Successful human-readable output includes:

- recipe schema;
- study name;
- canonical recipe digest.

It does not print prompt text.

For structural machine-readable metadata:

```bash
PYTHONPATH=src python -m metria recipe validate study.json --json
```

The JSON output contains the study name, run count, runtime names, measurement
names, analysis names, schema, and digest. It intentionally does not reproduce
measurement configs or prompt contents.

Validation failures return a non-zero exit status with a concise error on
stderr rather than a Python traceback.

## Digest

```bash
PYTHONPATH=src python -m metria recipe digest study.json
```

This prints only the canonical SHA-256 digest from `study_recipe_digest()`.
The digest identifies requested serialized intent; it is not proof that a
runtime applied the request.

## Normalize

```bash
PYTHONPATH=src python -m metria recipe normalize study.json
```

or:

```bash
PYTHONPATH=src python -m metria recipe normalize study.json \
  --output normalized-study.json
```

`normalize` validates the recipe and emits deterministic human-readable JSON.

### Privacy warning

Unlike `validate` and `digest`, **normalize reproduces the complete recipe**.
If measurement configs contain raw prompts, system messages, private paths, or
other sensitive inputs, normalized output contains them too.

Do not pipe normalized private recipes into public logs or publish them under
`artifacts/` unless those inputs are intentionally public.

## Why execution is not here yet

Study execution requires explicit runtime, measurement, and pairwise-analysis
registries. Adding automatic plugin discovery at the same time as the first CLI
would silently introduce a code-loading policy and make the command harder to
audit.

The safer sequence is:

1. stabilize recipe parsing and digesting;
2. stabilize this non-executing CLI surface;
3. package the root Metria core;
4. add explicit built-in registry selection;
5. add `metria run` only after its execution/provenance semantics are clear.
