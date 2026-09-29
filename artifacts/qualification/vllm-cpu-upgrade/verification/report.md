# Metria Verification

**Verdict: PASS**

Recipe: `c13fad665c1e7fc9d4ea9f93bd1db5cd4c2a520c7b8221e54d95c8e0a378cdf7`
Scope: local vLLM CPU runtime-stack upgrade

## Change:
  vLLM runtime environment: 0.29.0+cpu -> 0.30.0+cpu; controlled CPU IDs: [0, 1]

## Evidence:
  reference: completed (reference.run.json)
    Observed runtime: 0.29.0+cpu; context: 256
  candidate: completed (candidate.run.json)
    Observed runtime: 0.30.0+cpu; context: 256
  Lifecycle: completed
  measurements: matched
  model: matched
  observed.hardware: matched
  observed.worker_placement: matched
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
  metria.verification_impact: completed
    Task checks: reference 4/4, candidate 4/4 passed.
    Checked workload: 2/2 unique prompts; results describe declared checks only.
    Reference repeatability: 0/2 repeat pairs differed across 2 prompts.
    Observed sample repeatability; not a statistical guarantee or a quality score.
    Token differences are behavioral drift; task checks determine the declared quality outcomes.
  Loaded-engine request latency: -0.507214s (improved)
    Relative change: -46.9271%
    Sequential synchronous request calls after engine warmup; includes prompt evaluation and generation, excludes engine startup and cache reset. No streaming TTFT, serving-throughput, or statistical speedup claim.
  TTFT, decode throughput, inter-token latency, and device/KV memory: unavailable.

## Verdict:
  PASS
  Policy: PASS

Policy checks:
  quality.candidate_pass_rate@1 >= 1.0: PASS (observed 1.0 fraction)
  PASS means the user-defined criteria were met; it is not a universal deployment-safety judgment.
