# Metria Verification

**Verdict: VERIFIED**

Recipe: `dc7dea11e1058690aef1754fd163062ea871b447b1655dab265e92d20428337a`
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
      Category &quot;shared-prefix&quot;: 0/4 diverged
  metria.verification_impact: completed
    Task quality unavailable: complete declared task checks were not retained for both runs.
    Reference repeatability: 0/2 repeat pairs differed across 2 prompts.
    Observed sample repeatability; not a statistical guarantee or a quality score.
    Token differences are behavioral drift; task checks determine the declared quality outcomes.
  Loaded-engine request latency: -0.130286s (improved)
    Relative change: -13.7116%
    Sequential synchronous request calls after engine warmup; includes prompt evaluation and generation, excludes engine startup and cache reset. No streaming TTFT, serving-throughput, or statistical speedup claim.
  TTFT, decode throughput, inter-token latency, and device/KV memory: unavailable.

## Verdict:
  VERIFIED
  VERIFIED means comparison and analysis completed within the stated scope.
  No task-quality or performance acceptance policy was evaluated.
