# Metria Verification

**Verdict: PASS**

Recipe: `c14749ccb1e5494ae1edda9ec6e37dbe22c4c9a8d2bc7692cb576f69e6b29791`
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
    Divergent prompts: 0/24
    Trajectory evidence: complete; unavailable prompts: 0
    Length mismatches: 0
  metria.verification_impact: completed
    Task checks: reference 24/24, candidate 24/24 passed.
    Checked workload: 12/12 unique prompts; results describe declared checks only.
    Reference repeatability: 0/12 repeat pairs differed across 12 prompts.
    Observed sample repeatability; not a statistical guarantee or a quality score.
    Token differences are behavioral drift; task checks determine the declared quality outcomes.
  Client-side request latency (includes network): -1.13397s (improved)
    Relative change: -28.0296%
    Client-side wall time per sequential request, including the network, load balancer and any concurrent traffic on the server. Descriptive only; not a policy target.
  TTFT, decode throughput, inter-token latency, and device/KV memory: unavailable.

## Verdict:
  PASS
  Policy: PASS

Policy checks:
  quality.candidate_pass_rate_lower_bound@1 >= 0.5: PASS (observed 0.8620237953250198 fraction)
  quality.pass_rate_delta_lower_bound@1 >= -0.5: PASS (observed -0.13797620467498017 fraction_delta)
  behavior.trajectory_agreement@0.3.4 >= 1.0: PASS (observed 1.0 fraction)
  PASS means the user-defined criteria were met; it is not a universal deployment-safety judgment.
