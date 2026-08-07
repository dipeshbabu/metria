# KV Fidelity package identity

## Selected distribution

This repository uses KV Fidelity consistently across its distribution,
component, import, command, configuration, and report contracts:

```text
Distribution: kv-fidelity
Import:       kv_fidelity
Command:      kv-fidelity
Environment:  KV_FIDELITY_*
Report schema: kv_fidelity.report.*
```

Package-index availability is not a reservation or ownership guarantee. A
maintainer must configure and verify the pending Trusted Publisher described
below before treating the public package name as secured.

The source tree is versioned as `0.3.5.dev0`. The first planned stable release
under this identity is `0.3.5`, after release blockers
[#8](https://github.com/dipeshbabu/metria/issues/8),
[#11](https://github.com/dipeshbabu/metria/issues/11), and
[#12](https://github.com/dipeshbabu/metria/issues/12) are closed.
The publishing workflow enforces those issue states.

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

## Trusted Publisher identity

The PyPI pending publisher and GitHub release environment must use these exact
values:

| Field | Value |
|---|---|
| PyPI project | `kv-fidelity` |
| GitHub owner | `dipeshbabu` |
| GitHub repository | `dipeshbabu/metria` |
| Workflow | `publish-kv-fidelity.yml` |
| GitHub environment | `pypi-kv-fidelity` |
| Allowed tag pattern | `kv-fidelity-v*` |

The environment requires a reviewer and restricts deployments to matching
tags. PyPI publication uses OIDC only; there is no API-token fallback. The
post-publish job checks the PyPI Integrity API for the exact repository,
workflow, and environment, then cryptographically verifies both the wheel and
source distribution.

The repository-side environment and PyPI-side Trusted Publisher settings must
be re-verified immediately before the first release. Do not rely on a dated
configuration snapshot as proof that current release controls are still
correct.

## Ownership continuity

The first release is blocked until the exact PyPI account or organization
ownership and recovery access are documented. Use a PyPI organization account
or keep at least two trusted people as project Owners. Each owner must use
two-factor authentication and retain independent recovery codes. Removing an
owner requires confirming that two independent recovery paths remain.

GitHub reviewer access and PyPI ownership are separate. Adding a GitHub
environment reviewer does not provide PyPI recovery, and adding a PyPI Owner
does not grant repository access.
