# Metria execution and built-in registries

Metria's first execution CLI uses an **explicit built-in registry**. It does not
scan arbitrary Python entry points, import user-selected modules, install
inference engines, or execute code named by a recipe.

The current commands are:

```bash
metria plugins
metria plugins --json

metria run study.json --output-dir results/run-001
metria run study.json --output-dir results/run-001 --json
```

## Built-in registry

The root package currently registers:

### Runtimes

- `llamacpp`
- `vllm`

### Measurement protocols

- `kv_fidelity.decode_time_trajectory` (`0.3.4`)

### Pairwise analyses

- `kv_fidelity.trajectory_match` (`0.3.4`)

`metria plugins` reports one of three static availability states:

- `available` — the implementation/dependency needed by the registry is present;
- `recipe_dependent` — availability cannot be decided without recipe-specific
  paths/settings;
- `unavailable` — a required optional dependency is absent.

For example, vLLM can be marked unavailable when the optional `vllm` Python
package is not installed. llama.cpp is recipe-dependent because the current
adapter uses recipe-local binary/model paths.

Static registry availability is not the same as runtime support. After registry
validation, the normal Metria lifecycle still evaluates shared capability
rules and calls `RuntimeAdapter.probe()` for recipe-specific support evidence.

## No arbitrary plugin loading

Recipes contain implementation **names**, not Python import paths:

```json
{
  "runtime": {"name": "vllm"},
  "measurements": ["kv_fidelity.decode_time_trajectory"]
}
```

The first execution CLI resolves those names only through the built-in registry.
A recipe cannot make Metria import arbitrary code by naming a module or entry
point.

Third-party plugin loading, if added later, needs a separate trust/loading
policy and should not silently extend this command.

## Execute a recipe

Choose a new or empty result directory:

```bash
metria run study.json --output-dir results/experiment-a
```

The directory must be empty before execution. Metria refuses to overwrite an
existing evidence bundle.

Before launching any runtime, the CLI validates:

1. every requested runtime is registered;
2. every measurement is registered;
3. every requested pairwise analysis is registered;
4. statically unavailable implementations are rejected.

Recipe-dependent runtime checks still happen through the shared capability and
runtime preflight layers.

## Durable output

A completed invocation writes:

```text
results/experiment-a/
  run-0000.json
  run-0001.json
  ...
  study-result.json
```

Each run file is `metria.run_record.v1` and retains the same requested → resolved
→ observed evidence model used by the Python API.

The study manifest uses:

```text
metria.study_result.v1
```

It records:

- study name;
- canonical recipe schema/digest;
- captured hardware/software fingerprint;
- each run's path, lifecycle status, full-record digest, and evidence digest;
- pairwise compatibility reports;
- pairwise analysis status/results.

Run paths inside the manifest are relative file names so otherwise identical
runs in different output directories do not receive different manifests merely
because the parent directory changed.

## Invocation provenance

The recipe runner passes shared invocation provenance into every `RunRecord`:

```text
provenance.invocation
  recipe
    schema
    digest
  hardware
  orchestrator
    name
    version
```

This evidence is separate from `RunSpec.environment_selector` and the recipe's
requested `environment`. Requested placement is not promoted into observed host
identity.

The current stdlib hardware fingerprint intentionally does not claim accelerator
identity. Runtime adapters may add authoritative accelerator evidence through
their observed state.

## Failure preservation

An experimental failure does not erase the study.

If a runtime or measurement produces a normal Metria failure outcome,
`execute_study()` keeps the resulting `RunRecord`; `metria run` writes it to the
output directory and includes its status in the manifest.

CLI exit status is:

- `0` — all runs completed, all generated pairs are directly compatible, and
  every requested pairwise analysis completed;
- `1` — the study produced durable output but includes a failed/partial/timed-out
  run, incompatible pair, or failed/skipped analysis;
- `2` — invalid recipe/configuration, unavailable/unregistered implementation,
  non-empty output directory, or another command error before a durable study
  result is produced.

A nonzero experiment result is therefore distinguishable from invalid CLI input.

## Output determinism

With the same recipe, fixed hardware evidence, deterministic runtime/measurement
outputs, and the same Metria implementation, the generated manifest is
deterministic. Output-directory location is not embedded into the manifest's
run references.

This does **not** imply model inference itself is deterministic. Random seeds,
runtime scheduling, sampling configuration, hardware behavior, and upstream
engine versions remain experimental variables/evidence and must be controlled or
recorded by the study.

## Python API

The same orchestration boundary is available without the CLI:

```python
from metria import builtin_registry, execute_recipe_to_directory, load_study_recipe

recipe = load_study_recipe("study.json")
result = execute_recipe_to_directory(
    recipe,
    output_dir="results/experiment-a",
    registry=builtin_registry(),
)

print(result.successful)
print(result.manifest_path)
```

Passing a registry explicitly keeps orchestration testable and makes the set of
executable implementations visible rather than relying on ambient imports.
