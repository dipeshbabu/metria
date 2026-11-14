# Metria Verification

**Verdict: PASS**

Recipe: `294dcf082db4cc2c6d58b032611bf9cb5f932ad041ed6bc131725d37d3bdaca5`
Scope: STAGED remote endpoint comparison; server identity was reported by the server, not verified by Metria

## Change:
  Declared endpoint configuration: {'dtype': 'float32'} -> {'dtype': 'bfloat16'}

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
    Token prefix agreement: 95.3125/100
    Exact token-sequence matches: 75%
    Divergent prompts: 2/8
    Trajectory evidence: complete; unavailable prompts: 0
    First divergence (zero-based): earliest 13, median 13.0
    Length mismatches: 0
      "0:story": first divergence at token 13
      "1:story": first divergence at token 13
  metria.verification_impact: completed
    Task checks: reference 8/8, candidate 8/8 passed.
    Checked workload: 4/4 unique prompts; results describe declared checks only.
    Reference repeatability: 0/4 repeat pairs differed across 4 prompts.
    Observed sample repeatability; not a statistical guarantee or a quality score.
    Token differences are behavioral drift; task checks determine the declared quality outcomes.
  metria.trajectory_kld: completed
    Divergent prompts: 8
      0:story: first divergence at token 13
      0:dog: first divergence at token None
      0:capital: first divergence at token None
      0:arithmetic: first divergence at token None
      1:story: first divergence at token 13
      1:dog: first divergence at token None
      1:capital: first divergence at token None
      1:arithmetic: first divergence at token None
  Client-side request latency (includes network): +0.347197s (regressed)
    Relative change: +43.2802%
    Client-side wall time per sequential request, including the network, load balancer and any concurrent traffic on the server. Descriptive only; not a policy target.
  TTFT, decode throughput, inter-token latency, and device/KV memory: unavailable.

## Verdict:
  PASS
  Policy: PASS

Policy checks:
  quality.candidate_pass_rate@1 >= 1.0: PASS (observed 1.0 fraction)
  behavior.reference_repeatability@1 >= 1.0: PASS (observed 1.0 fraction)
  behavior.trajectory_agreement@0.3.4 >= 0.9: PASS (observed 0.953125 fraction)
  behavior.trajectory_kld_mean@0.1.0 [staged] <= 0.1: PASS (observed 0.05592205681797503 nats)
  PASS means the user-defined criteria were met; it is not a universal deployment-safety judgment.
