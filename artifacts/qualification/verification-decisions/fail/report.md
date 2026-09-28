# Metria Verification

**Verdict: FAIL**

Recipe: `8b3f55cb5b731a75844be9d02fa45a0c343f0982e7425093435db88e8bf21abb`
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
    Task checks: reference 4/8, candidate 4/8 passed.
    Checked workload: 2/2 unique prompts; results describe declared checks only.
      Failed &quot;deliberate-failure&quot; on &quot;0:shared-a&quot; (exact_text); answer content retained by digest.
      Failed &quot;deliberate-failure&quot; on &quot;0:shared-b&quot; (exact_text); answer content retained by digest.
      Failed &quot;deliberate-failure&quot; on &quot;1:shared-a&quot; (exact_text); answer content retained by digest.
      Failed &quot;deliberate-failure&quot; on &quot;1:shared-b&quot; (exact_text); answer content retained by digest.
    Reference repeatability: 0/2 repeat pairs differed across 2 prompts.
    Observed sample repeatability; not a statistical guarantee or a quality score.
    Token differences are behavioral drift; task checks determine the declared quality outcomes.
  Loaded-engine request latency: -0.322901s (improved)
    Relative change: -27.2505%
    Sequential synchronous request calls after engine warmup; includes prompt evaluation and generation, excludes engine startup and cache reset. No streaming TTFT, serving-throughput, or statistical speedup claim.
  TTFT, decode throughput, inter-token latency, and device/KV memory: unavailable.

## Verdict:
  FAIL
  Policy: FAIL

Policy checks:
  quality.candidate_pass_rate@1 >= 1.0: FAIL (observed 0.5 fraction)
  PASS means the user-defined criteria were met; it is not a universal deployment-safety judgment.
