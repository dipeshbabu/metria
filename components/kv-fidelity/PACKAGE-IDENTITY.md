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

The `kv-fidelity` name returned 404 from PyPI's project JSON API on
2026-07-25. That check does not reserve the name. A maintainer must create the
pending Trusted Publisher described below before treating the name as secured.

The source tree is versioned as `0.3.5.dev0`. The first planned stable release
under this identity is `0.3.5`, after release blockers
[#8](https://github.com/dipeshbabu/efficient-llm-systems/issues/8),
[#11](https://github.com/dipeshbabu/efficient-llm-systems/issues/11), and
[#12](https://github.com/dipeshbabu/efficient-llm-systems/issues/12) are closed.
The publishing workflow enforces those issue states.

## Legacy `refract-llm` releases

PyPI's `refract-llm` releases through 0.3.2.3 were uploaded by a legacy
maintainer with MIT, author, and repository metadata that do not describe this
repository. They contain an earlier codebase from this project's development
ancestry, but they were not released from
`dipeshbabu/efficient-llm-systems` and are not Apache-2.0 releases of this
repository.

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
| GitHub repository | `dipeshbabu/efficient-llm-systems` |
| Workflow | `publish-kv-fidelity.yml` |
| GitHub environment | `pypi-kv-fidelity` |
| Allowed tag pattern | `kv-fidelity-v*` |

The environment requires a reviewer and restricts deployments to matching
tags. PyPI publication uses OIDC only; there is no API-token fallback. The
post-publish job checks the PyPI Integrity API for the exact repository,
workflow, and environment, then cryptographically verifies both the wheel and
source distribution.

As of 2026-07-25, the GitHub environment exists with `@dipeshbabu` as required
reviewer and the `kv-fidelity-v*` tag policy. Administrator bypass is still
enabled and must be disabled in the GitHub environment settings before a
release. The pending PyPI Trusted Publisher has not yet been created.

## Ownership continuity

The first release is blocked until the exact PyPI account or organization
ownership and recovery access are documented. Use a PyPI organization account
or keep at least two trusted people as project Owners. Each owner must use
two-factor authentication and retain independent recovery codes. Removing an
owner requires confirming that two independent recovery paths remain.

GitHub reviewer access and PyPI ownership are separate. Adding a GitHub
environment reviewer does not provide PyPI recovery, and adding a PyPI Owner
does not grant repository access.
