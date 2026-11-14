# Verification acceptance policies

Metria 0.1.2 supports optional, user-defined acceptance
policies. Published Metria 0.1.0 reports `VERIFIED` when comparison and analysis
complete; it does not evaluate these policies. Use the current workspace for this
feature until the next root release is published.

A policy answers whether a valid reference/candidate comparison meets your
criteria. Metria provides no universal behavior or safety threshold. `PASS`
means only that the criteria you supplied were met.

Add an optional top-level `policy` to an existing verification recipe:

```json
{
  "schema": "metria.verification_policy.v1",
  "criteria": [
    {
      "target": "behavior.trajectory_agreement",
      "version": "0.3.4",
      "unit": "fraction",
      "min": 0.98
    },
    {
      "target": "behavior.full_match_rate",
      "version": "0.3.4",
      "unit": "fraction",
      "min": 0.90
    }
  ]
}
```

The numbers above are illustrative user choices, not recommended deployment
thresholds. The fragment is the value of `recipe.policy`; keep the recipe's
existing `metria.study_recipe.v1` schema and reference/candidate configuration.

From the repository root, run:

```bash
uv sync --locked --all-packages
uv run --locked --all-packages metria recipe validate study.json
uv run --locked --all-packages metria verify study.json --output policy-check
```

The output directory must be new. See the [local verifier guide](metria-verify.md)
for preparing the qualified CPU runtime, model, and recipe.

## Typed targets

Targets are registered identifiers, not arbitrary JSON paths. Each names the
analysis that produces it and that analysis's method version, which is not the
Metria package version. A target's analysis must be declared in the recipe, and
each verification profile declares only the analyses it supports.

**Behavior** (`kv_fidelity.trajectory_match`, version `0.3.4`):

| Target | Unit | Criteria |
|---|---|---|
| `behavior.trajectory_agreement` | `fraction` | Numeric `min` and/or `max`, within 0–1 |
| `behavior.full_match_rate` | `fraction` | Numeric `min` and/or `max`, within 0–1 |
| `behavior.divergence_rate` | `fraction` | Numeric `min` and/or `max`, within 0–1; lower is better |
| `behavior.all_trajectories_match` | `boolean` | Exact `equals: true` or `equals: false` |
| `analysis.status` | `status` | Exact `equals: "completed"`, `"failed"`, or `"skipped"` |

**Task quality, repeatability and latency** (`metria.verification_impact`,
version `1`). These need declared task checks or repeated reference trials; see
[task checks and repeatability](verification-decisions.md) and
[performance measurements](verification-performance.md).

| Target | Unit | Criteria |
|---|---|---|
| `quality.candidate_pass_rate` | `fraction` | Numeric bounds within 0–1 |
| `quality.reference_pass_rate` | `fraction` | Numeric bounds within 0–1 |
| `quality.pass_rate_delta` | `fraction_delta` | Candidate minus reference, within −1–1 |
| `behavior.reference_repeatability` | `fraction` | Numeric bounds within 0–1 |
| `quality.candidate_pass_rate_lower_bound` | `fraction` | Lower 95% Wilson bound of the candidate pass rate |
| `quality.pass_rate_delta_lower_bound` | `fraction_delta` | Lower 95% Newcombe bound of candidate minus reference, within −1–1 |
| `quality.pass_rate_delta_upper_bound` | `fraction_delta` | Upper 95% Newcombe bound of candidate minus reference, within −1–1 |
| `performance.candidate_latency_seconds` | `seconds` | Nonnegative; lower is better |
| `performance.latency_ratio` | `ratio` | Candidate/reference, nonnegative; lower is better |

**Serving** (`metria.serving_impact`, version `1`, from
[serving measurements](serving-measurements.md)). Each metric has a
`_candidate` absolute target and a `_ratio` candidate/reference target, all
nonnegative:

| Metric | Candidate unit | Better |
|---|---|---|
| `serving.ttft_seconds_*` | `seconds` | lower |
| `serving.inter_token_latency_seconds_*` | `seconds` | lower |
| `serving.request_latency_seconds_*` | `seconds` | lower |
| `serving.output_tokens_per_second_*` | `tokens/second` | higher |
| `serving.decode_tokens_per_second_*` | `tokens/second` | higher |
| `serving.requests_per_second_*` | `requests/second` | higher |
| `serving.allocated_kv_bytes_*` | `bytes` | lower |
| `serving.peak_device_allocated_bytes_*` | `bytes` | lower |
| `serving.peak_device_reserved_bytes_*` | `bytes` | lower |

Ratio targets use unit `ratio`.

**Batch invariance and client load** (`metria.batch_invariance`, version
`1`), available only in [endpoint verification](endpoint-verification.md#concurrency-and-batch-invariance)
with a declared `--concurrency`:

| Target | Unit | Criteria |
|---|---|---|
| `behavior.reference_batch_invariance` | `fraction` | Numeric bounds within 0–1 |
| `behavior.candidate_batch_invariance` | `fraction` | Numeric bounds within 0–1 |
| `client.requests_per_second_ratio` | `ratio` | Nonnegative; higher is better; client-side, includes the network |
| `client.request_latency_ratio` | `ratio` | Nonnegative; lower is better; client-side, includes the network |

**STAGED: trajectory KL divergence** (`metria.trajectory_kld`, version `0.1.0`).
Only the FP8 KV-cache profile accepts these, and only when prepared with
`--staged-kld`. Their report rows carry `"qualification": "staged"` and the
Markdown report marks them `[staged]`. See
[staged capabilities](staged-capabilities.md#trajectory-kl-divergence).

| Target | Unit | Criteria |
|---|---|---|
| `behavior.trajectory_kld_mean` | `nats` | Numeric `min` and/or `max`, at least 0; lower is better |
| `behavior.trajectory_kld_max` | `nats` | Numeric `min` and/or `max`, at least 0; lower is better |

The verifier still requires completed analysis before evaluating a policy.
Requesting `analysis.status == "failed"` cannot make a failed execution pass.

Trajectory agreement is the existing 0–100 score divided by 100. The report
retains both the source metric identity/unit and this explicit normalization.
Exact matches use the analysis's boolean result, without rounding a numeric
score into a boolean.

Each criterion must specify a target and version. The unit may be omitted and
is resolved from the typed catalog; an explicitly wrong unit is rejected.
Numeric criteria cannot use `equals`, and boolean/status criteria cannot use
numeric bounds. Empty policies, duplicate targets, contradictory min/max bounds,
unknown fields, unsupported versions, and non-finite thresholds are rejected
before execution. Latency targets use the verifier's own measured trials, never
the process timing display.

## Evaluation order and CI exits

Policy evaluation follows these gates:

```text
execution completed
  -> required identity and runtime evidence present
  -> comparison valid
  -> required analyses completed
  -> typed policy evidence available
  -> PASS or FAIL
```

`NOT_COMPARABLE`, `INSUFFICIENT_EVIDENCE`, and `EXECUTION_FAILED` never become
`PASS` through a policy. If an earlier gate fails, criteria are retained as
`NOT_EVALUATED`. If a required policy value is absent, incomplete, or has a
different unit/direction/method/version/aggregation, the outcome is
`INSUFFICIENT_EVIDENCE`. Missing values are not numeric zero. Every criterion is
retained, including observed values, source identity, and its result/reason.

| Exit | Meaning |
|---|---|
| `0` | `PASS`, or `VERIFIED` when no policy was supplied |
| `1` | Policy `FAIL` |
| `2` | Invalid input/configuration |
| `3` | `NOT_COMPARABLE` |
| `4` | `INSUFFICIENT_EVIDENCE` |
| `5` | Execution/preflight or persistence failure |
| `130` | Interrupted verification |

The manifest adds a `policy` evaluation with schema `metria.policy_evaluation.v1`
when a policy is present. The policy is also part of the canonical recipe digest.
Recipes without a policy retain their previous digest shape and `VERIFIED`
completion semantics. Python callers can construct `PolicyCriterion` and
`VerificationPolicy` objects, or use `policy_from_data`.
