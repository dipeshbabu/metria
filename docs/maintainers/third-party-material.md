# Third-party material and artifact provenance

Metria's root and component `LICENSE`/`NOTICE` files describe their original
software. They do not grant rights to redistribute someone else's model,
dataset, tokenizer, engine, or generated output. Keep upstream terms and source
identity with retained third-party material. A public download is not evidence
of redistribution permission.

| Material | Repository treatment |
|---|---|
| Original Metria/component code and documentation | Keep the owning package's license and notices; preserve attribution in derived code. |
| Vendored engine code or patches | Retain upstream license, immutable source revision, local patch identity, and modifications. |
| Model weights, tokenizers, and datasets | Download separately unless redistribution rights have been established. Retain upstream identifier, revision, digest, and license/source reference. |
| Generated text, model-derived arrays, and sampled tokens | Do not assign Apache-2.0, CC0, CC-BY, or another license automatically. Record applicable source terms, rights status, privacy constraints, and any explicit publication decision. |
| Measurements and reports | Preserve methodology and input provenance. Distinguish original explanatory text from third-party/model-derived content. |
| Historical evidence with incomplete sources | Label reproducibility gaps. Preserve negative and superseded conclusions. Do not invent missing licenses, revisions, or hardware facts. |

Maintainers review vendored material and redistribution decisions before release.
When rights are unknown, record `unknown` and retain only permissible metadata or
links until resolved. Package builds must include their own `LICENSE` and
`NOTICE`; source checks and wheel-content checks enforce this requirement.

## Shared representation

Use the existing `ArtifactManifest` primitive for downloaded inputs, generated
run evidence, and curated results. Its `source` and `metadata` mappings retain
provenance. `artifact_manifest_to_data()` and `artifact_manifest_from_data()`
provide the `metria.artifact_manifest.v1` envelope. JSON serialization is
deterministic. The older `artifact_to_data()` remains the same primitive without
an envelope for existing embedded run provenance.

## Headline artifacts

A headline artifact is a retained result explicitly cited as supporting current
guidance or a qualified support claim. Add its versioned manifest to
[`artifacts/headline-manifests.json`](../../artifacts/headline-manifests.json).
The first consumer wraps the existing llama.cpp CPU qualification report without
changing its original report, records, or measurements.

Every newly designated headline manifest requires:

- repository-relative artifact path, SHA-256, and byte size;
- code, runtime, model, and workload identifiers with immutable revisions or
  content hashes; a mutable branch or release label alone is insufficient;
- recipe/configuration digest, retained hardware and software evidence;
- upstream sources, license information (including explicit unknown status),
  and redistribution status;
- manifest creation timestamp, precise claim scope, and artifact rights status.

Keep command/configuration details in the referenced recipe or run record and
identify them by digest. `created_at` timestamps the manifest; retain the original
measurement date separately when curating an older result. Do not invent a time
for a historical measurement that only records a date. Container digests,
compiler identity, upstream notices, and source archive URIs belong in the
corresponding source mapping when relevant.

`validate_headline_manifest()` checks required provenance. The repository checker
also checks retained file hashes and sizes. Validation establishes neither the
truth of an asserted identity nor redistribution permission. A license marked
unknown remains unknown.

## Historical migration

Historical cleanup uses this same representation and records gaps explicitly.
An archival result is not promoted into the headline index just to pass a check.
Use immutable source links or licensed patch archives when available. Document
unavailable forks and avoid claiming their results are independently reproducible.
