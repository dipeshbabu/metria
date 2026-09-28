# Metria Verification

**Verdict: VERIFIED**

Recipe: `05d650a2fb49eed30c822c04c6dab486ea226f29123bc9963f6aa3313d50c458`
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
    Divergent prompts: 0/6
    Trajectory evidence: complete; unavailable prompts: 0
    Length mismatches: 0
      Category &quot;shared-prefix&quot;: 0/6 diverged
  Loaded-engine request latency: -0.0361885s (improved)
    Relative change: -5.75456%
    Sequential synchronous request calls after engine warmup; includes prompt evaluation and generation, excludes engine startup and cache reset. No streaming TTFT, serving-throughput, or statistical speedup claim.
  TTFT, decode throughput, inter-token latency, and device/KV memory: unavailable.

## Verdict:
  VERIFIED
  VERIFIED means comparison and analysis completed within the stated scope.
  No task-quality or performance acceptance policy was evaluated.
