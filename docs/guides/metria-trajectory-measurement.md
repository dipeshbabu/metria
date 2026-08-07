# Metria trajectory measurement bridge

Metria's first KV Fidelity bridge preserves an important distinction between
**run evidence** and **pairwise fidelity metrics**.

A single inference run can produce a decode-time model-token trajectory. It
cannot, by itself, have a "trajectory agreement" score. Agreement exists only
when that run is compared with a reference run under a valid study comparison
plan.

## Run-local capture

`TokenTrajectoryProtocol` requests semantic `token_ids` evidence from the
runtime session:

```python
from metria.measurements import TokenTrajectoryProtocol

protocol = TokenTrajectoryProtocol()
result = protocol.execute(
    session,
    scenario={},
    config={
        "prompts": [
            {"id": "p1", "prompt": "The capital of France is"},
            {"id": "p2", "prompt": "2 + 2 ="},
        ],
        "generation": {"temperature": 0.0},
    },
)
```

The resulting `MeasurementResult` contains:

- descriptive metrics such as mean captured trajectory length;
- immutable per-prompt token-ID evidence;
- prompt SHA-256 fingerprints and prompt IDs;
- no raw prompt text in retained measurement evidence.

The prepared runtime may still receive the prompt text during inference. The
measurement result intentionally stores only the fingerprint needed to verify
that two runs evaluated the same prompt.

## Pairwise trajectory agreement

After obtaining a reference result and a candidate result:

```python
from metria.measurements import compare_trajectory_results

comparison = compare_trajectory_results(reference_result, candidate_result)
score = comparison.metrics["trajectory_agreement_score"].value
```

The comparison reproduces the KV Fidelity v0.3.4 trajectory rule:

```text
score = 100 * sum(prefix agreement steps)
              / sum(max(reference steps, candidate steps))
```

The denominator is computed per prompt before aggregation. This means a
candidate that stops early and a candidate that continues after the reference
are both penalized against the longer trajectory.

Before computing a score, Metria verifies:

- both results use the same trajectory capture schema;
- both use the same method and method version;
- prompt-ID sets are identical;
- the SHA-256 prompt fingerprint for every ID matches;
- captured token-ID data has the expected structure.

If both trajectories are empty for the same prompt, the pair is not scored. An
empty decode-time capture cannot provide enough evidence to distinguish an
immediate stop from missing or incompatible instrumentation.

## Relationship to KV Fidelity

The algorithm is intentionally compatible with the current KV Fidelity
trajectory methodology, but the implementation does not call
`kv_fidelity.runner` or its module-global backend selector. Runtime execution is
owned by Metria's `RuntimeSession`; the measurement layer only asks for semantic
`token_ids` capture.

This keeps the standalone `kv-fidelity` package unchanged while allowing the
methodology to migrate into Metria's run/evidence model incrementally.

## Current limitations

This bridge does not yet:

- orchestrate reference and candidate runs automatically;
- attach measurement results to a `RunRecord` through a common executor;
- provide trajectory KLD/logit capture;
- provide a vLLM or SGLang Metria runtime adapter;
- expose a CLI recipe for paired fidelity studies.

The next orchestration step should execute a `RunSpec` through a runtime adapter
and measurement protocol, then produce a complete `RunRecord` with measurement
metrics/evidence and observed runtime provenance.
