# Typed identity and evidence primitives

Metria's experiment model remains **Study → Run → Evidence**.

The typed primitives in `metria.identity` do not introduce a second experiment
object graph. They provide validated constructors and reusable evidence objects
at boundaries that were previously represented only by unstructured mappings.

## Relationship to the study core

```text
StudySpec
  └── RunSpec
      ├── model      ← ModelRef or an equivalent mapping
      ├── runtime    ← RuntimeConfig or an equivalent mapping
      ├── scenario   ← WorkloadSpec or an equivalent mapping
      ├── treatments
      └── measurements

RunRecord
  ├── requested
  ├── resolved
  ├── observed      ← may include HardwareFingerprint-shaped evidence
  ├── metrics
  ├── evidence
  └── artifacts     ← may include ArtifactManifest
```

`ModelRef`, `RuntimeConfig`, and `WorkloadSpec` implement Python's `Mapping`
interface. `RunSpec` therefore deep-freezes them through the same path used for
ordinary dictionaries. The retained `RunSpec` representation remains a mapping,
so existing runtime adapters do not need a parallel typed code path.

This also preserves `metria.study_recipe.v1`: a typed constructor and the
semantically equivalent dictionary produce the same recipe data and canonical
digest.

## Requested identity is not observed identity

A `ModelRef` records requested model and tokenizer identity. Fields such as
`revision` or `geometry` are not proof that a launched server actually served
those values.

The normal evidence sequence remains:

```text
requested → resolved → observed
```

Runtime adapters are responsible for resolving requested identity and collecting
whatever authoritative post-launch evidence their runtime can expose. Missing
observed evidence should remain unknown rather than being inferred from the
request.

## ModelRef

`ModelRef` requires at least a model identifier or a local path. It can retain:

- model id and revision;
- local path;
- tokenizer id and revision;
- model geometry supplied by an inspector or caller;
- additional metadata.

Geometry is evidence available to later capability checks. Merely storing
`head_dim`, KV-head count, or an architecture label does not itself mark an
optimization as supported.

## RuntimeConfig and WorkloadSpec

`RuntimeConfig` requires a non-empty runtime name and optionally a version. Its
`config` fields are flattened into the existing runtime mapping, which preserves
adapter compatibility:

```python
RuntimeConfig(
    name="vllm",
    config={"dtype": "bfloat16", "kv_cache_dtype": "fp8"},
)
```

normalizes to the same requested mapping as:

```python
{
    "name": "vllm",
    "dtype": "bfloat16",
    "kv_cache_dtype": "fp8",
}
```

`WorkloadSpec` follows the same pattern for the `RunSpec.scenario` mapping.
Reserved identity keys cannot be redefined through extension configuration.

## Capability states

Capability discovery uses four conservative states:

- `supported` — the implemented rule has sufficient evidence to support the
  requested capability;
- `experimental` — the capability is intentionally available but outside the
  normal validated support envelope;
- `unsupported` — available evidence proves the request is not supported;
- `unknown` — evidence is missing, contradictory, or insufficient.

`unknown` is distinct from `unsupported`. An adapter should not turn missing
metadata into a confident incompatibility claim, and it should not turn a user
request into proof of support.

`SupportReport` uses the same `SupportLevel` vocabulary so runtime preflight and
future `metria inspect` functionality can converge on one meaning.

## HardwareFingerprint

`HardwareFingerprint` is structured **observed evidence**, not a placement
request. It can retain platform, host, accelerator, software, and additional
metadata while remaining deeply immutable.

The initial type deliberately does not define a GPU-family support matrix or
claim that a hardware name determines performance. Capability rules and measured
results should consume the fingerprint as evidence.

## ArtifactManifest

`ArtifactManifest` provides a shared identity/provenance shape for models,
datasets, reports, generated outputs, patches, containers, or other retained
artifacts. It can represent:

- artifact name and kind;
- URI and/or local path;
- immutable revision;
- SHA-256 digest;
- byte size;
- upstream source/license information;
- additional provenance metadata.

The type validates SHA-256 syntax and byte sizes but does not claim that a hash
was independently verified. Downloaders and resolvers remain responsible for
computing and checking digests before constructing verified provenance.

## What these types do not solve yet

This layer does not yet provide:

- automatic model-geometry inspection;
- runtime capability rules;
- server-side model/tokenizer verification;
- automatic hardware capture;
- versioned `RunRecord` serialization;
- artifact downloading or safe archive extraction;
- a root `metria` distribution release.

Those are follow-on consumers of this shared vocabulary. In particular, new
runtime, benchmark, evaluator, and provenance work should reuse these primitives
instead of introducing another incompatible identity schema.
