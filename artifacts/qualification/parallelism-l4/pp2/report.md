# Metria Verification

**Verdict: PASS**

Recipe: `6ab70e873e24996047dd8c701a34830dfb4c41cc96bc853bd40e2b5a0475ac3f`
Scope: STAGED local vLLM tensor/pipeline parallelism comparison

## Change:
  Parallelism: {'pipeline_parallel_size': 1, 'tensor_parallel_size': 1} -> {'pipeline_parallel_size': 2, 'tensor_parallel_size': 1}

## Evidence:
  reference: completed (reference.run.json)
    Observed tensor parallel: 1; pipeline parallel: 1; visible devices: 2
  candidate: completed (candidate.run.json)
    Observed tensor parallel: 1; pipeline parallel: 2; visible devices: 2
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
    Divergent prompts: 0/8
    Trajectory evidence: complete; unavailable prompts: 0
    Length mismatches: 0
  metria.verification_impact: completed
    Task checks: reference 8/8, candidate 8/8 passed.
    Checked workload: 4/4 unique prompts; results describe declared checks only.
    Reference repeatability: 0/4 repeat pairs differed across 4 prompts.
    Observed sample repeatability; not a statistical guarantee or a quality score.
    Token differences are behavioral drift; task checks determine the declared quality outcomes.
  metria.trajectory_kld: completed
    Divergent prompts: 8
      0:story: first divergence at token None
      0:dog: first divergence at token None
      0:capital: first divergence at token None
      0:arithmetic: first divergence at token None
      1:story: first divergence at token None
      1:dog: first divergence at token None
      1:capital: first divergence at token None
      1:arithmetic: first divergence at token None
  Loaded-engine request latency: -0.0931684s (improved)
    Relative change: -36.2998%
    Sequential synchronous request calls after engine warmup; includes prompt evaluation and generation, excludes engine startup and cache reset. No streaming TTFT, serving-throughput, or statistical speedup claim.
  TTFT, decode throughput, inter-token latency, and device/KV memory: unavailable.

## Verdict:
  PASS
  Policy: PASS

Policy checks:
  quality.pass_rate_delta@1 >= 0.0: PASS (observed 0.0 fraction_delta)
  behavior.trajectory_agreement@0.3.4 >= 0.9: PASS (observed 1.0 fraction)
  behavior.trajectory_kld_mean@0.1.0 [staged] <= 0.05: PASS (observed 0.0 nats)
  PASS means the user-defined criteria were met; it is not a universal deployment-safety judgment.
