# Metria Verification

**Verdict: PASS**

Recipe: `0212b24f5b6ccd15b0bb2f411c01210039096c5bd83851d039f038666351a9a0`
Scope: local vLLM streaming serving API; no HTTP/network timing

## Change:
  Client concurrency: 1 -> 2; fixed engine capacity: 2

## Evidence:
  reference: completed (reference.run.json)
    Observed runtime: 0.30.0+cpu; context: 256
  candidate: completed (candidate.run.json)
    Observed runtime: 0.30.0+cpu; context: 256
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
  Local serving request latency: +1.53024s (regressed)
    Relative change: +98.7447%
    Local AsyncLLM serving API after warmup; excludes engine startup, resets and HTTP/network overhead. Single-device observations; no statistical speedup claim.
    request_latency_seconds: 1.54969 -> 3.07993 seconds
    ttft_seconds: 0.0946992 -> 0.252866 seconds
    inter_token_latency_seconds: 0.0969976 -> 0.18847 seconds
    decode_tokens_per_second: 10.3095 -> 5.30589 tokens/second
    output_tokens_per_second: 10.3242 -> 10.3892 tokens/second
    requests_per_second: 0.645261 -> 0.649323 requests/second
    peak_device_allocated_bytes: unavailable
    peak_device_reserved_bytes: unavailable
    allocated_kv_bytes: unavailable
    Memory: native worker PyTorch allocator peaks and allocated KV tensor storage; excludes other processes and non-PyTorch allocations.
    Decode rate: generated tokens after the first divided by summed per-request stream intervals; unavailable when token chunks coalesce.

## Verdict:
  PASS
  Policy: PASS

Policy checks:
  quality.candidate_pass_rate@1 >= 1.0: PASS (observed 1.0 fraction)
  PASS means the user-defined criteria were met; it is not a universal deployment-safety judgment.
