# KV Fidelity within Metria

KV Fidelity is Metria's behavioral measurement methodology. Its implementation
and bundled assets ship in the public `metria` distribution, using
`metria.fidelity` Python APIs and the `metria fidelity` expert command group.
The primary reference/candidate workflow remains `metria verify`.

The standalone `kv-fidelity` publication plan is retired. This source directory
retains characterization tests, method documentation, licenses, and a local
compatibility bridge. It is not published separately and does not require a
separate PyPI account, publisher, tag, or release recovery arrangement.

Old `kv_fidelity` imports and `kv-fidelity` commands are supported by the bridge
inside a full uv workspace. New installations use Metria. Existing
`kv_fidelity.report.*` schemas, `KV_FIDELITY_*` environment configuration, method
names and method versions remain intact for scientific/source compatibility.
The historical fidelity framework version identifies those reports; the Metria
distribution has its own release version. See the
[migration guide](../../docs/guides/unified-fidelity.md).

## Legacy `refract-llm` releases

PyPI's `refract-llm` releases through 0.3.2.3 were uploaded by a legacy
maintainer with MIT, author, and repository metadata that do not describe this
repository. They contain an earlier codebase from this project's development
ancestry, but they were not released from `dipeshbabu/metria` and are not
Apache-2.0 releases of this repository.

Do not install `refract-llm` as a substitute for this source tree, and do not
rewrite its historical PyPI metadata. KV Fidelity deliberately uses a
different import package and command, so new integrations do not depend on the
legacy project's public names.
