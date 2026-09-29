# Metria Verification

**Verdict: PASS**

Recipe: `6765fb7667589192fc787d941dad5f3d1b43e14b3544813947757107b2b1de87`
Scope: local vLLM streaming serving API; no HTTP/network timing

## Change:
  Client concurrency: 1 -> 2; fixed engine capacity: 2

## Evidence:
  reference: completed (reference.run.json)
    Observed runtime: 0.30.0+cu129; context: 256
  candidate: completed (candidate.run.json)
    Observed runtime: 0.30.0+cu129; context: 256
  Lifecycle: completed
  measurements: matched
  model: matched
  runtime: matched
  scenario: matched
  trial_policy.measured_trials: matched
  trial_policy.method: matched
  trial_policy.warmup_trials: matched

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
  metria.serving_impact: completed
  Local serving request latency: +0.0464808s (regressed)
    Relative change: +4.9491%
    Local AsyncLLM serving API after warmup; excludes engine startup, resets and HTTP/network overhead. Single-device observations; no statistical speedup claim.
    request_latency_seconds: 0.939178 -> 0.985659 seconds
    ttft_seconds: 0.135272 -> 0.161313 seconds
    inter_token_latency_seconds: 0.0535918 -> 0.0549544 seconds
    decode_tokens_per_second: 18.6596 -> 18.1969 tokens/second
    output_tokens_per_second: 17.0351 -> 32.4182 tokens/second
    requests_per_second: 1.06469 -> 2.02614 requests/second
    peak_device_allocated_bytes: 1.46158e+09 -> 1.46168e+09 bytes
    peak_device_reserved_bytes: 1.46801e+09 -> 1.46801e+09 bytes
    allocated_kv_bytes: 1.17522e+09 -> 1.17522e+09 bytes
    Memory: native worker PyTorch allocator peaks and allocated KV tensor storage; excludes other processes and non-PyTorch allocations.
    Decode rate: generated tokens after the first divided by summed per-request stream intervals; unavailable when token chunks coalesce.

## Verdict:
  PASS
  Policy: PASS

Policy checks:
  quality.candidate_pass_rate@1 >= 1.0: PASS (observed 1.0 fraction)
  serving.allocated_kv_bytes_candidate@1 >= 1.0: PASS (observed 1175224320.0 bytes)
  PASS means the user-defined criteria were met; it is not a universal deployment-safety judgment.
