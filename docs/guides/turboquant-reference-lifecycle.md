# TurboQuant Reference source lifecycle

TurboQuant Reference remains a source-only research/reference component. It is
independently versioned and locally buildable, with distribution name
`turboquant-reference` and import `turboquant`. It is not folded into root Metria
and does not define a second public Metria workflow.

The previous planned package-index publishing workflow is retired. Root Metria owns the protected publication procedure and includes KV Fidelity
methods in its distribution. Publishing
this research component later requires a new reviewed lifecycle decision,
upstream-rights review, namespace/publisher verification, and an end-to-end release
rehearsal; this document makes no package-index ownership or availability claim.

## Pin and validate a source snapshot

Use a full immutable Metria commit, retain it with the component version and
environment, and install the component subdirectory. The
[component README](../../components/turboquant-reference/README.md) contains an
exact source-install example. Its project URLs identify the Metria repository
and component path.

In the checked-out revision:

```bash
uv sync --locked --all-packages
uv run python -m pytest components/turboquant-reference/tests
uv run python -m build --outdir dist/turboquant components/turboquant-reference
uv run python -m twine check dist/turboquant/*
uv run python tools/maintenance/check_wheel_contents.py dist/turboquant/*.whl
uv run python components/turboquant-reference/benchmarks/examples/demo.py
```

CI continues to test the component across its supported Python/platform matrix,
build its wheel and source distribution, inspect metadata/licenses/contents,
install the built wheel, and run the demo. Source-only status does not remove
these checks. Python support is declared in its own manifest; currently 3.10–3.14.

The component maintainer listed in [MAINTAINERS.md](../../MAINTAINERS.md) owns
its changelog, source version, compatibility notes, and reproducibility guidance.
During the alpha series, source compatibility remains provisional; a full commit
pin and recorded dependency environment identify the implementation actually used.

Follow the [third-party material policy](../maintainers/third-party-material.md)
for upstream references and generated evidence. New headline results use the
shared `ArtifactManifest`, including source/input/runtime identities and rights
metadata. Local builds do not grant redistribution rights to external models or
datasets, and no generated/model-derived material is automatically relicensed.
