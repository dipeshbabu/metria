# Verify one inference change

Start with a deterministic fixture to learn the report, then prepare the qualified
local llama.cpp recipe for a real comparison. A fixture success is not hardware,
runtime, model-quality, or deployment qualification.

## Complete flow without a model or GPU

From this repository's current development checkout:

```bash
uv sync --locked --all-packages
uv run python examples/verification/run_fixture.py --case pass --output fixture-pass
uv run python examples/verification/run_fixture.py --case fail --output fixture-fail
uv run python examples/verification/run_fixture.py --case not-comparable --output fixture-invalid
```

The commands return 0, 1, and 3 respectively. Use new output directories. They run
the real recipe/CLI/execution/comparison/policy/persistence flow, replacing only
the external subprocess and hardware boundary with explicit test evidence.
Every fixture report is labeled synthetic and its canonical JSON has
`fixture_only: true`. Never cite these outputs as real engine evidence.

Reference requests one CPU thread; candidate requests two. Both retain matching
synthetic model/provider/workload evidence and identical generation settings.
The candidate differs at the third sampled token. The PASS/FAIL cases use
illustrative user thresholds of 0.6 and 0.9; neither is a recommended threshold.
The invalid case introduces an undeclared vocabulary change, so the comparison
fails before the policy can approve it.

Each directory contains `verification.json`, `report.md`, and both run records.
The report shape is:

```text
Metria Verification: PASS / FAIL / NOT_COMPARABLE
Scope: synthetic fixture

CHANGE     reference threads 1 -> candidate threads 2
EVIDENCE   model/provider/workload controls and observed readback
COMPARISON valid, or the unexpected vocabulary difference
IMPACT     trajectory prefix agreement and divergence diagnostics
VERDICT    explicit policy result, or comparison failure
```

## Real llama.cpp configuration regression

Follow the [qualified runtime/model setup](../../docs/guides/metria-verify.md).
The preparation tool writes a copyable recipe with explicit reference/candidate
order, model content hash, and capture-provider hash:

```bash
python tools/qualification/prepare_cpu_verification.py \
  --bin-dir /path/to/qualified/build/bin --model stories260K.gguf \
  --reference-threads 1 --candidate-threads 2 --output thread-change.json
metria recipe validate thread-change.json
metria verify thread-change.json --output thread-verification --json
```

This also demonstrates a supported inference-configuration change. Metria checks
that the requested threads actually applied, keeps other controls fixed, compares
behavior, and reports compatible cold-process request latency. Add an explicit
[acceptance policy](../../docs/guides/verification-policies.md) if you want PASS/FAIL;
without one, successful comparison means VERIFIED.

## Staged common-change templates

The [catalog](catalog.json) records readiness and blockers. The versioned recipes
under `staged/` validate structurally, but the current `metria verify` CLI rejects
their unsupported routes. They are planning templates, not runnable support claims.

| Change | Reference / candidate | Required evidence before enabling |
|---|---|---|
| Runtime upgrade | pinned vLLM build A / pinned build B | Separate qualified environments, immutable model/tokenizer, matching generation/workload/hardware, and a verifier route that executes both builds. |
| KV-cache precision | auto cache / FP8 cache | Qualified hardware/engine support plus authoritative applied-cache readback and compatible systems measurement. |
| Quantization/inference treatment | reference configuration / declared candidate treatment | Qualified treatment application and unchanged unrelated controls; no guessed or relabeled precision. |
| llama.cpp build regression | old qualified capture provider / new provider | Independent provider hashes/qualification and a verifier scope permitting the binary change. The runnable thread example above covers configuration changes today. |

Do not enable a template merely because a fixture passes or a runtime imports.
The public CLI stays narrow until the new route meets its evidence contract.
The source recipes and readiness catalog are validated in CI. Use the
[CI workflow example](github-actions.yml) to retain a real verification result
and summary after preparing an appropriate runner and recipe.
