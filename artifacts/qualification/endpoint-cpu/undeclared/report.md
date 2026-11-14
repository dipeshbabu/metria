# Metria Verification

**Verdict: NOT_COMPARABLE**

Recipe: `9c4e3a1d6bf25d99eb62c911296ca9995be8a5bae1a2a728c7f499b8ac9faf1c`
Scope: STAGED remote endpoint comparison; server identity was reported by the server, not verified by Metria

## Change:
  Declared endpoint configuration: {'dtype': 'float32'} -> {'dtype': 'float32'}

## Evidence:
  reference: completed (reference.run.json)
    Claimed runtime: 0.30.0; context: 256; identity: partial (claimed)
  candidate: completed (candidate.run.json)
    Claimed runtime: 0.30.0; context: 512; identity: partial (claimed)
  Lifecycle: completed
  measurements: matched
  model: matched
  scenario: matched
  trial_policy: matched

## Comparison:
  NOT_COMPARABLE
  observed.identity.model.max_model_len: controlled dimension differs
  observed.identity_snapshots.after.model.max_model_len: undeclared comparison-relevant difference
  observed.identity_snapshots.before.model.max_model_len: undeclared comparison-relevant difference

## Impact:
  kv_fidelity.trajectory_match: skipped
  metria.verification_impact: skipped
    Decision evidence unavailable: methodology is missing or incompatible.
  metria.trajectory_kld: skipped
  Performance impact unavailable: reference/candidate comparison gate did not pass
  TTFT, decode throughput, inter-token latency, and device/KV memory: unavailable.
  No candidate benefit is inferred from an invalid or incomplete comparison.

## Verdict:
  NOT_COMPARABLE
  Policy: NOT_EVALUATED

Policy checks:
  quality.candidate_pass_rate@1 >= 1.0: NOT_EVALUATED (observed unavailable fraction)
    verification gate: NOT_COMPARABLE
  behavior.reference_repeatability@1 >= 1.0: NOT_EVALUATED (observed unavailable fraction)
    verification gate: NOT_COMPARABLE
  behavior.trajectory_agreement@0.3.4 >= 1.0: NOT_EVALUATED (observed unavailable fraction)
    verification gate: NOT_COMPARABLE
  behavior.trajectory_kld_mean@0.1.0 [staged] <= 1e-09: NOT_EVALUATED (observed unavailable nats)
    verification gate: NOT_COMPARABLE
  PASS means the user-defined criteria were met; it is not a universal deployment-safety judgment.
