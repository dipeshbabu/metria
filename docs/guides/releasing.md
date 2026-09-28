# Releasing Python packages

Before publication, follow the [third-party material and provenance policy](../maintainers/third-party-material.md)
and run `python tools/maintenance/check_provenance.py` from the repository root.

Package publication is a maintainer-only operation. Build artifacts are created
in a job without publishing credentials, then a separate protected job uploads
the verified artifacts with PyPI Trusted Publishing. No PyPI token is stored in
GitHub.

For the root `metria` distribution, follow the separate
[Metria release procedure](releasing-metria.md). The component identities and
release blockers below apply to their respective packages.

Before creating a KV Fidelity release tag, verify that its Metria dependency can
be installed from the public index:

```bash
uv run --locked python tools/maintenance/check_public_dependencies.py components/kv-fidelity/pyproject.toml
```

As checked on 2026-09-28, the public Metria release is 0.1.0 while KV Fidelity
requires `metria>=0.1.1.dev0,<0.2`. KV Fidelity publication is therefore blocked
until a compatible stable Metria release is published and verified. A workspace
install cannot substitute for that prerequisite. Keep the component development
version until its release can be validated end to end.

The publication workflow requires issues #8, #11, #12, and #40 to be closed and
checks public dependency availability before building/uploading. Its clean-wheel
install resolves dependencies from the public index. Do not bypass these guards
or create a stable release tag merely to make a blocked publication proceed.

## KV Fidelity package identity

This repository publishes KV Fidelity as `kv-fidelity`, with Python import
`kv_fidelity` and CLI command `kv-fidelity`. PyPI's legacy `refract-llm`
project is controlled by a legacy maintainer and its releases were not
published from this repository. See
[the package-identity record](../../components/kv-fidelity/PACKAGE-IDENTITY.md)
before changing release or ownership settings.

No `kv-fidelity` release exists yet. Until the first release is verified,
user documentation must direct users to a source checkout instead of claiming
that a package-index version is available. The first planned stable version is
0.3.5.

### One-time setup for `kv-fidelity`

1. In PyPI, create a pending Trusted Publisher for project `kv-fidelity` with:
   - GitHub owner: `dipeshbabu`
   - repository: `metria`
   - workflow: `publish-kv-fidelity.yml`
   - environment: `pypi-kv-fidelity`
2. Create the GitHub environment `pypi-kv-fidelity`. Require a reviewer,
   disallow administrator bypass, and restrict deployments to tags matching
   `kv-fidelity-v*`.
3. Establish two independent PyPI recovery paths before the first release.
   Prefer a PyPI organization account; otherwise keep at least two trusted
   people as project Owners, with two-factor authentication and separate
   recovery codes.
4. Confirm that release-blocking issues #8, #11, #12, and #40 are closed. The
   workflow also checks them and refuses to build while any is open.
5. Keep the GitHub environment and Trusted Publisher values exact. The
   workflow has no password or long-lived token fallback.

### `kv-fidelity` release procedure

1. In a pull request, change the development version in
   `components/kv-fidelity/pyproject.toml` and
   `components/kv-fidelity/src/kv_fidelity/__init__.py` to the stable release version.
   Finalize the matching dated section in
   `components/kv-fidelity/CHANGELOG.md`. The first planned release is 0.3.5.
2. Merge only after the required CI and security checks pass on the exact
   release commit.
3. Tag the merge commit as `kv-fidelity-v<VERSION>` and push the tag. The tag
   must point to a commit reachable from `main`.
4. Dispatch the publishing workflow from that exact tag:

   ```bash
   gh workflow run publish-kv-fidelity.yml \
     --ref kv-fidelity-v0.3.5 \
     -f version=0.3.5
   ```

5. Review the clean-wheel smoke results and artifact hashes, then approve the
   protected `pypi-kv-fidelity` deployment.
6. Let the workflow finish. After upload, it installs the exact version from
   PyPI in a clean environment; verifies distribution metadata, Apache-2.0
   license files, project links, `kv-fidelity --version`, and bundled prompts;
   checks the exact Trusted Publisher identity through PyPI's Integrity API;
   cryptographically verifies the wheel and source-distribution attestations;
   and creates the matching GitHub release.

PyPI does not permit replacing files for an existing version. If publication
succeeds but a later verification or GitHub release step fails, do not rerun
the publish job. Repair the post-publish step against the existing PyPI files
and tag.

## TurboQuant Reference source lifecycle

TurboQuant Reference is source-only research software. It retains an independent
source version, component-local wheel/sdist builds, supported Python versions,
license files, tests, and demo checks. There is no package-index release workflow
or release-tag contract. Follow the [source lifecycle](turboquant-reference-lifecycle.md)
to install and cite an immutable repository revision.
