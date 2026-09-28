# Releasing Metria and maintaining source components

Metria is the public package and release unit. It includes the KV Fidelity
implementation, prompt assets, and report readers under `metria.fidelity`.
Follow the [Metria release procedure](releasing-metria.md) for protected builds,
tag/CI validation, Trusted Publishing, clean installs, and signed provenance.

The former standalone KV Fidelity publication plan in
[issue #13](https://github.com/dipeshbabu/metria/issues/13) is superseded by
[unification #146](https://github.com/dipeshbabu/metria/issues/146). There is no
separate KV Fidelity upload or tag to create. Its local compatibility bridge is
not a publishable product. Keep source/report compatibility and attribution
according to the [migration guide](unified-fidelity.md) and
[package identity record](../../components/kv-fidelity/PACKAGE-IDENTITY.md).

TurboQuant Reference remains source-only and uses
[immutable Git revisions](turboquant-reference-lifecycle.md). CI continues to
build and test local distributions and their licenses/assets. A source-only
component does not require a PyPI publisher or deployment approval.

All releases follow the [third-party material and provenance policy](../maintainers/third-party-material.md).
Preserve historical results and component method versions independently of the
root distribution's version. A new release must not silently reinterpret old
report schemas or relabel research evidence as current qualification.
