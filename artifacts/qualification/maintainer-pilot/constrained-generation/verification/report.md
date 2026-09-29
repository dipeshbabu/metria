# Metria Verification

**Verdict: FAIL**

Recipe: `9603a8622f2c98221b27fe0c574be0caf9c9126d11969e761b4409f4fc10aed1`
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
    Task checks: reference 2/12, candidate 2/12 passed.
    Checked workload: 6/6 unique prompts; results describe declared checks only.
      Failed &quot;sentiment&quot; on &quot;0:positive&quot; (exact_text); answer content retained by digest.
      Failed &quot;sentiment&quot; on &quot;0:negative&quot; (exact_text); answer content retained by digest.
      Failed &quot;order&quot; on &quot;0:order-id&quot; (exact_text); answer content retained by digest.
      Failed &quot;shape&quot; on &quot;0:status-json&quot; (json_object); answer content retained by digest.
      Failed &quot;shape&quot; on &quot;0:ticket-json&quot; (json_object); answer content retained by digest.
      Failed &quot;sentiment&quot; on &quot;1:positive&quot; (exact_text); answer content retained by digest.
      Failed &quot;sentiment&quot; on &quot;1:negative&quot; (exact_text); answer content retained by digest.
      Failed &quot;order&quot; on &quot;1:order-id&quot; (exact_text); answer content retained by digest.
      Failed &quot;shape&quot; on &quot;1:status-json&quot; (json_object); answer content retained by digest.
      Failed &quot;shape&quot; on &quot;1:ticket-json&quot; (json_object); answer content retained by digest.
    Reference repeatability: 0/6 repeat pairs differed across 6 prompts.
    Observed sample repeatability; not a statistical guarantee or a quality score.
    Token differences are behavioral drift; task checks determine the declared quality outcomes.
  Loaded-engine request latency: -0.126574s (improved)
    Relative change: -49.0788%
    Sequential synchronous request calls after engine warmup; includes prompt evaluation and generation, excludes engine startup and cache reset. No streaming TTFT, serving-throughput, or statistical speedup claim.
  TTFT, decode throughput, inter-token latency, and device/KV memory: unavailable.

## Verdict:
  FAIL
  Policy: FAIL

Policy checks:
  quality.reference_pass_rate@1 >= 1.0: FAIL (observed 0.16666666666666666 fraction)
  quality.candidate_pass_rate@1 >= 1.0: FAIL (observed 0.16666666666666666 fraction)
  PASS means the user-defined criteria were met; it is not a universal deployment-safety judgment.
