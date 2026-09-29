# Metria Verification

**Verdict: FAIL**

Recipe: `454643419d784d280b2e55a5be26519c9ad8d4a16f9ec7a01f9c5b8961ed23cf`
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
    Divergent prompts: 0/12
    Trajectory evidence: complete; unavailable prompts: 0
    Length mismatches: 0
      Category &quot;classification&quot;: 0/4 diverged
      Category &quot;extraction&quot;: 0/4 diverged
      Category &quot;structured-output&quot;: 0/4 diverged
  metria.verification_impact: completed
    Task checks: reference 6/12, candidate 6/12 passed.
    Checked workload: 6/6 unique prompts; results describe declared checks only.
      Failed &quot;sentiment&quot; on &quot;0:positive&quot; (exact_text); answer content retained by digest.
      Failed &quot;sentiment&quot; on &quot;0:negative&quot; (exact_text); answer content retained by digest.
      Failed &quot;order&quot; on &quot;0:order-id&quot; (exact_text); answer content retained by digest.
      Failed &quot;sentiment&quot; on &quot;1:positive&quot; (exact_text); answer content retained by digest.
      Failed &quot;sentiment&quot; on &quot;1:negative&quot; (exact_text); answer content retained by digest.
      Failed &quot;order&quot; on &quot;1:order-id&quot; (exact_text); answer content retained by digest.
    Reference repeatability: 0/6 repeat pairs differed across 6 prompts.
    Observed sample repeatability; not a statistical guarantee or a quality score.
    Token differences are behavioral drift; task checks determine the declared quality outcomes.
  Loaded-engine request latency: +1.30256s (regressed)
    Relative change: +152.14%
    Sequential synchronous request calls after engine warmup; includes prompt evaluation and generation, excludes engine startup and cache reset. No streaming TTFT, serving-throughput, or statistical speedup claim.
  TTFT, decode throughput, inter-token latency, and device/KV memory: unavailable.

## Verdict:
  FAIL
  Policy: FAIL

Policy checks:
  quality.reference_pass_rate@1 >= 1.0: FAIL (observed 0.5 fraction)
  quality.candidate_pass_rate@1 >= 1.0: FAIL (observed 0.5 fraction)
  PASS means the user-defined criteria were met; it is not a universal deployment-safety judgment.
