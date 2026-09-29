# Metria Verification

**Verdict: PASS**

Recipe: `6aac97c1bf34a2f7311fd75a4d5f276e0a6c0f51ad382b4526e8cdd7e1b2cee9`
Scope: local vLLM prefix-cache comparison

## Change:
  Prefix caching: False -> True

## Evidence:
  reference: completed (reference.run.json)
    Observed prefix caching: False; context: 512
  candidate: completed (candidate.run.json)
    Observed prefix caching: True; context: 512
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
    Divergent prompts: 0/4
    Trajectory evidence: complete; unavailable prompts: 0
    Length mismatches: 0
      Category &quot;literal&quot;: 0/4 diverged
  metria.verification_impact: completed
    Task checks: reference 4/4, candidate 4/4 passed.
    Checked workload: 2/2 unique prompts; results describe declared checks only.
    Reference repeatability: 0/2 repeat pairs differed across 2 prompts.
    Observed sample repeatability; not a statistical guarantee or a quality score.
    Token differences are behavioral drift; task checks determine the declared quality outcomes.
  Loaded-engine request latency: -0.0626879s (improved)
    Relative change: -18.5144%
    Sequential synchronous request calls after engine warmup; includes prompt evaluation and generation, excludes engine startup and cache reset. No streaming TTFT, serving-throughput, or statistical speedup claim.
  TTFT, decode throughput, inter-token latency, and device/KV memory: unavailable.

## Verdict:
  PASS
  Policy: PASS

Policy checks:
  quality.reference_pass_rate@1 >= 1.0: PASS (observed 1.0 fraction)
  quality.candidate_pass_rate@1 >= 1.0: PASS (observed 1.0 fraction)
  PASS means the user-defined criteria were met; it is not a universal deployment-safety judgment.
