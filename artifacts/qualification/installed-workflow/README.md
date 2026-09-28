# Installed-package workflow qualification

This maintainer-run walkthrough installed the Metria wheel built from source
`fa9484f73d257b7f3243bd535e07dcb987d48cd9` into the pinned vLLM CPU environment.
It unset `PYTHONPATH`, changed directory to `/var/tmp`, and used the actual
installed console scripts to prepare the bundled example, validate the recipe,
and run `metria verify --json`.

The CLI JSON matched the saved canonical result. Both runs completed, the
comparison was valid, and the result was [VERIFIED](report.md). The example
descriptor/workload came from installed package resources, not checkout-relative
files. `qualification.json` records the wheel/source identity and command flow.

Original run records, recipe and reports are retained with the shared provenance
manifest. The same pinned model/tokenizer/runtime environment and custom dependency
limitations as the [CPU prefix qualification](../vllm-prefix-cache/README.md) apply.
No model or runtime binaries are redistributed, and no additional license is
asserted for model-derived output tokens.

This is implementation and usability-path evidence from the maintainer. It is not
external-user feedback, a model-quality benchmark or a production speedup claim.
Use the [pilot protocol](../../../docs/guides/verifier-pilot.md) to collect actual
participant feedback separately.
