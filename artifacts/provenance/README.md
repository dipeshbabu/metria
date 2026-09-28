# Retained evidence and source mapping

Current support claims use the [headline manifest index](../headline-manifests.json)
and the shared `metria.artifact_manifest.v1` representation. Historical records
with missing source/input evidence remain archival and are not promoted into that
index. The same representation records their known identity and explicit gaps.

| Retained result family | Best available source/input/environment evidence | Reproducibility limits |
|---|---|---|
| Local llama.cpp CPU thread qualification | [Shared manifest](../qualification/llamacpp-cpu-threads/headline-manifest.json), [pinned capture patch](../../tools/qualification/llamacpp-capture.patch), model hash, recipe, run records, hardware/software and binary hashes | Tiny-model CPU integration only; original report retains its scope and date. |
| Legacy dense/MoE KLD and perplexity logs | [Logs](../benchmarks/legacy-raw/README.md), per-file shared manifests in `legacy-raw/`, model basenames and configuration embedded in logs | Experimental fork source archive and immutable model hashes are missing; numerical outputs alone cannot reproduce the experiment. |
| Bundled KV Fidelity report examples | [Examples](../../components/kv-fidelity/src/kv_fidelity/examples/README.md), per-file shared manifests in `kv-fidelity-examples/` | Illustrations of report interpretation; incomplete historical model/runtime identity does not establish current qualified support. |
| Sparse-V/threshold ablations | [Research investigations](../../research/investigations/threshold-ablation.md) and raw output in `artifacts/ablations/sparse-v-threshold/` | Historical fork availability and incomplete input/environment identity constrain reproduction; negative and superseded results remain intact. |
| NIAH records | [Quality investigations](../../research/investigations/quality-benchmarks.md) and dated JSON/Markdown in `artifacts/niah/` | Consult each record's model/configuration fields; filenames or timestamps are not immutable model/runtime identity. |
| MLX quality records | [MLX guide](../../docs/guides/mlx-port.md) and dated reports in `artifacts/mlx/` | Historical experimental fork source is not replaced by an upstream MLX checkout. |
| EDEN scale investigation | [Paper](../../research/papers/eden-optimal-s-revisit.md) and [immutable upstream EDEN source](https://github.com/amitport/EDEN-Distributed-Mean-Estimation/tree/5c7639a6af810e08d21827dd0ed55772b8113e99) | The public EDEN side is pinned; local TurboQuant and engine-fork sides retain the gaps stated in the paper. |

See [historical fork availability](../../docs/reference/historical-forks.md).
No unavailable fork was reconstructed by substituting upstream code. The existing
llama.cpp capture patch retains its upstream MIT license; other immutable public
references are linked without copying code whose redistribution was not reviewed.

## Path-only redaction

Eight legacy logs previously contained a contributor's home-directory prefix.
Their current copies replace that prefix with `${HOME}`. Model basenames,
numerical values, flags, and all remaining canonical Git-blob bytes are unchanged.
Each `.artifact.json` records the original Git revision and SHA-256, the redacted
file's SHA-256/size, and the exact class of edit. Git history preserves the original
record. Current log checkouts use LF so digest validation agrees across platforms.

The same prefix-only operation applies to four bundled KV Fidelity JSON examples
and their HTML renderings, with separate original/redacted manifests. Corpus
content hashes and measured values are unchanged; only display/configuration
paths use the portable placeholder.

Two paper references now use historical checkout-relative paths. The MI300X
deployment directory uses `${EXPERIMENT_WORKSPACE}`. These edits remove workstation
locations without changing claims or results. New tracked documents and evidence
are checked for common macOS/Linux/Windows home paths; portable placeholders are
allowed. There are currently no blanket archival exclusions from that check.
