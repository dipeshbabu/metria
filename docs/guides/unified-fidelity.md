# Fidelity methods in one Metria installation

The Metria 0.2 development line includes KV Fidelity's implementation, bundled
prompts and report examples. Users install one public distribution and use one
CLI. The previously published Metria 0.1.1 retains its original command surface;
use this checkout until the new root release is published.

```bash
uv sync --locked --all-packages
uv run metria verify --help
uv run metria fidelity --help
uv run metria fidelity score --help
uv run metria fidelity compare --help
```

`metria verify` is the primary reference/candidate workflow, with explicit
identity, lifecycle, comparison, policy, and retained evidence. `metria fidelity`
provides the existing expert scoring, self-test, comparison, repeatability, and
corpus-fetch commands. Those commands preserve their established behavior;
bundling them does not qualify new verifier runtimes, all scoring axes, or old
experimental engine forks. Legacy composite bands are diagnostic conventions,
not deployment approval or a universal quality guarantee.

## Python API and assets

```python
from importlib import resources
from metria.fidelity.contracts import CompletionResult, KLDResult
from metria.fidelity.measurement_math import approximate_topk_kl

prompts = resources.files("metria.fidelity").joinpath("prompts/v0.1.jsonl")
```

Modules under `metria.fidelity.axes`, `metria.fidelity.backends`, and the existing
report/comparison/math modules are available from the same distribution. Engine
libraries remain lazy and optional. Root extras `mlx`, `sglang`, `research`, and
`benchmarks` select their dependencies; installing base Metria installs no
inference stack. The `full` extra retains the previously managed backend set;
it does not imply every runtime can share one compatible environment.

## Existing source users and reports

The local `components/kv-fidelity` project is a source-only compatibility bridge.
In the uv workspace, old `kv_fidelity` module imports resolve to the canonical
implementation and the old `kv-fidelity` command remains available during
migration. New code uses `metria.fidelity`; installed root wheels expose the
`metria` command. This bridge is not a separate PyPI product.

Saved `kv_fidelity.report.*` schemas, method names/versions, environment variables,
and numerical behavior remain unchanged. Existing numerical and CLI suites test
the same implementation through the compatibility imports. Report assets moved
without rewriting their evidence bytes; archival manifests retain their content
hashes and point to the new paths. Licensing and historical attribution remain.

The standalone KV Fidelity publishing workflow is retired. Metria's protected
release procedure owns package publication, signatures and install verification.
TurboQuant Reference keeps its source-only research lifecycle.
