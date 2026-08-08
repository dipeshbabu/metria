# Metria study recipes

Metria recipes are versioned, portable descriptions of **requested study
intent**. The first recipe format is JSON and uses this schema identifier:

```text
metria.study_recipe.v1
```

The recipe layer deliberately serializes configuration, not live Python
implementations. Runtime adapters, measurement protocols, and pairwise analysis
plugins are registered separately when a recipe is executed.

## Why JSON first

The first recipe format uses only the Python standard library and adds no root
runtime dependency to Metria. JSON also gives the project one unambiguous schema
and canonical byte representation before a YAML frontend is added.

YAML can later be supported as an input syntax that parses into this same
versioned schema. It should not become a second semantic format.

## Recipe contents

A recipe contains:

```text
schema
study
  name
  runs[]
    model
    runtime
    scenario
    measurements[]
    treatments[]
    trial_policy
    environment_selector
  comparison
    vary[]
    control[]
    block_by[]
    analyses[]
  constants
  metadata
measurement_configs
environment
```

`measurement_configs` contains method-specific requested input, such as the
prompt set for `TokenTrajectoryProtocol`. `environment` is the shared requested
environment mapping used by the current study executor.

Neither field is observed runtime evidence.

## Loading and writing

```python
from metria import (
    dump_study_recipe,
    load_study_recipe,
    study_recipe_digest,
)

recipe = load_study_recipe("study.json")
print(study_recipe_digest(recipe))

dump_study_recipe("normalized-study.json", recipe)
```

`dump_study_recipe()` writes deterministic UTF-8 JSON with sorted object keys.
Comparison dimension sets are emitted in sorted order while ordered analysis
names preserve study order.

## Canonical digest

`study_recipe_digest()` computes SHA-256 over a compact canonical JSON
representation:

```python
digest = study_recipe_digest(recipe)
```

Mappings with different insertion order therefore receive the same digest when
their requested semantics are identical.

The digest identifies the serialized request. It is **not** a result hash, model
hash, runtime hash, or proof that the requested configuration was actually
applied. Requested, resolved, and observed state remain separate.

Automatic attachment of the recipe digest to execution provenance is a later
orchestration step; the current recipe API exposes the digest without claiming
that linkage already exists.

## Shared RunSpec schema

The requested run representation is shared with `metria.run_record.v1` through
`run_spec_to_data()` and `run_spec_from_data()`. A saved run record therefore
retains the exact same requested RunSpec semantics as the study recipe that
planned it; result serialization does not invent a parallel request schema.

See [run records and comparison](metria-run-records.md) for durable result
serialization and `metria compare`.

## Strict validation

The v1 parser fails on:

- unknown schema versions;
- unknown object fields;
- missing required fields;
- invalid treatment kinds;
- malformed arrays or mappings;
- duplicate comparison analyses;
- measurement configs for measurements not requested by the study.

The serializer also rejects programmatic values that cannot be represented by
strict JSON, including arbitrary live Python objects, sets, bytes, and
non-finite floating-point values.

This fail-loud behavior is intentional. Silently dropping an unknown field can
change the experiment while leaving the recipe looking valid.

## Recipes can contain sensitive inputs

Run evidence redacts prompt/system text where possible, but a recipe is an
**input file** and may need to contain the actual prompts required to run a
measurement.

For example:

```json
{
  "measurement_configs": {
    "kv_fidelity.decode_time_trajectory": {
      "prompts": [
        {"id": "p1", "prompt": "actual prompt text"}
      ]
    }
  }
}
```

Do not assume a study recipe is safe to publish merely because Metria's retained
runtime evidence is privacy-conscious. Public benchmark recipes can contain
public prompts; private or proprietary prompt sets should be handled as private
inputs.

## Plugin references

Recipes name runtime, measurement, and pairwise-analysis implementations but do
not serialize Python objects or import paths.

For example:

```json
{
  "runtime": {"name": "vllm"},
  "measurements": ["kv_fidelity.decode_time_trajectory"]
}
```

and:

```json
{
  "analyses": ["kv_fidelity.trajectory_match"]
}
```

Execution must supply registries containing implementations with exactly those
names. This keeps recipes data-only and avoids turning a recipe file into an
arbitrary code-loading mechanism.

## Current limits

The recipe/result layer still does not provide:

- YAML parsing;
- plugin auto-discovery;
- environment placement across multiple hosts;
- recipe locking against downloaded model artifacts;
- automatic recipe-digest attachment to every `RunRecord`;
- a public `metria run` command.

The CLI does provide recipe validation/digest/normalization, capability
inspection, and comparison of already-saved versioned run records. The next
execution CLI should build on these schemas rather than bypassing them.
