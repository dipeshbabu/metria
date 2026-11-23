# Metria Verification

**Verdict: PASS**

Recipe: `100bbc25efaca6e21b24f680c326090f940a51a8f4f43a3bfaf5f628de5ade90`
Scope: STAGED remote endpoint comparison; server identity was reported by the server, not verified by Metria

## Change:
  Declared endpoint configuration: {'engine_version': '0.30.0', 'kv_cache_dtype': 'auto'} -> {'engine_version': '0.30.0', 'kv_cache_dtype': 'fp8'}

## Evidence:
  reference: completed (reference.run.json)
    Claimed runtime: 0.30.0; context: 512; identity: partial (claimed)
  candidate: completed (candidate.run.json)
    Claimed runtime: 0.30.0; context: 512; identity: partial (claimed)
  Lifecycle: completed
  measurements: matched
  model: matched
  scenario: matched
  trial_policy: matched

## Comparison:
  VALID

## Impact:
  kv_fidelity.trajectory_match: completed
    Token prefix agreement: 48.4375/100
    Exact token-sequence matches: 25%
    Divergent prompts: 6/8
    Trajectory evidence: complete; unavailable prompts: 0
    First divergence (zero-based): earliest 1, median 1.0
    Length mismatches: 0
      "0:arithmetic": first divergence at token 1
      "0:capital": first divergence at token 1
      "1:arithmetic": first divergence at token 1
      "1:capital": first divergence at token 1
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
      0:capital: first divergence at token 1
      0:arithmetic: first divergence at token 1
      1:story: first divergence at token 13
      1:dog: first divergence at token None
      1:capital: first divergence at token 1
      1:arithmetic: first divergence at token 1
  Client-side request latency (includes network): -0.0394513s (improved)
    Relative change: -10.943%
    Client-side wall time per sequential request, including the network, load balancer and any concurrent traffic on the server. Descriptive only; not a policy target.
  TTFT, decode throughput, inter-token latency, and device/KV memory: unavailable.

## Verdict:
  PASS
  Policy: PASS

Policy checks:
  quality.candidate_pass_rate@1 >= 1.0: PASS (observed 1.0 fraction)
  behavior.trajectory_kld_mean@0.1.0 [staged] <= 0.5: PASS (observed 0.10496912001587588 nats)
  PASS means the user-defined criteria were met; it is not a universal deployment-safety judgment.
