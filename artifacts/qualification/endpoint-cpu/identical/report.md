# Metria Verification

**Verdict: PASS**

Recipe: `9c4e3a1d6bf25d99eb62c911296ca9995be8a5bae1a2a728c7f499b8ac9faf1c`
Scope: STAGED remote endpoint comparison; server identity was reported by the server, not verified by Metria

## Change:
  Declared endpoint configuration: {'dtype': 'float32'} -> {'dtype': 'float32'}

## Evidence:
  reference: completed (reference.run.json)
    Claimed runtime: 0.30.0; context: 256; identity: partial (claimed)
  candidate: completed (candidate.run.json)
    Claimed runtime: 0.30.0; context: 256; identity: partial (claimed)
  Lifecycle: completed
  measurements: matched
  model: matched
  scenario: matched
  trial_policy: matched

## Comparison:
  VALID

## Impact:
  kv_fidelity.trajectory_match: completed
    Token prefix agreement: 100/100
    Exact token-sequence matches: 100%
    Divergent prompts: 0/8
    Trajectory evidence: complete; unavailable prompts: 0
    Length mismatches: 0
  metria.verification_impact: completed
    Task checks: reference 8/8, candidate 8/8 passed.
    Checked workload: 4/4 unique prompts; results describe declared checks only.
    Reference repeatability: 0/4 repeat pairs differed across 4 prompts.
    Observed sample repeatability; not a statistical guarantee or a quality score.
    Token differences are behavioral drift; task checks determine the declared quality outcomes.
  metria.trajectory_kld: completed
    Divergent prompts: 8
      0:story: first divergence at token None
      0:dog: first divergence at token None
      0:capital: first divergence at token None
      0:arithmetic: first divergence at token None
      1:story: first divergence at token None
      1:dog: first divergence at token None
      1:capital: first divergence at token None
      1:arithmetic: first divergence at token None
  Client-side request latency (includes network): -3.98838s (improved)
    Relative change: -65.4675%
    Client-side wall time per sequential request, including the network, load balancer and any concurrent traffic on the server. Descriptive only; not a policy target.
  TTFT, decode throughput, inter-token latency, and device/KV memory: unavailable.

## Verdict:
  PASS
  Policy: PASS

Policy checks:
  quality.candidate_pass_rate@1 >= 1.0: PASS (observed 1.0 fraction)
  behavior.reference_repeatability@1 >= 1.0: PASS (observed 1.0 fraction)
  behavior.trajectory_agreement@0.3.4 >= 1.0: PASS (observed 1.0 fraction)
  behavior.trajectory_kld_mean@0.1.0 [staged] <= 1e-09: PASS (observed 0.0 nats)
  PASS means the user-defined criteria were met; it is not a universal deployment-safety judgment.
