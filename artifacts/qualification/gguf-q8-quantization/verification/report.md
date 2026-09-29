# Metria Verification

**Verdict: PASS**

Recipe: `face5af93c5136e52dd3d8f702edbd0a8a491529eba5927100b5b4f914a6d4fb`
Scope: local llama.cpp GGUF weight-quantization comparison

## Change:
  Observed tensor storage: {'F32': 48} -> {'F16': 5, 'F32': 11, 'Q8_0': 32} (Q8_0 conversion; mixed storage retained)

## Evidence:
  reference: completed (reference.run.json)
    Observed threads: 2; context: 256
  candidate: completed (candidate.run.json)
    Observed threads: 2; context: 256
  Lifecycle: completed
  measurements: matched
  observed.gguf.controls_sha256: matched
  observed.gguf.tensor_layout_sha256: matched
  observed.gguf.tokenizer_sha256: matched
  runtime: matched
  scenario: matched

## Comparison:
  VALID

## Impact:
  kv_fidelity.trajectory_match: completed
    Token prefix agreement: 100/100
    Exact token-sequence matches: 100%
    Divergent prompts: 0/3
    Trajectory evidence: complete; unavailable prompts: 0
    Length mismatches: 0
  metria.verification_impact: completed
    Task checks: reference 3/3, candidate 3/3 passed.
    Checked workload: 3/3 unique prompts; results describe declared checks only.
    Reference variability unavailable: at least two retained reference trials per prompt are required.
    Token differences are behavioral drift; task checks determine the declared quality outcomes.
  reference mean process wall time: 0.0194621s (includes startup and model loading)
  candidate mean process wall time: 0.019024s (includes startup and model loading)
  Cold-process request latency: -0.000438033s (improved)
    Relative change: -2.2507%
    One cold invocation per prompt; no statistical speedup claim.
  TTFT, decode throughput, inter-token latency, and device/KV memory: unavailable.

## Verdict:
  PASS
  Policy: PASS

Policy checks:
  quality.candidate_pass_rate@1 >= 1.0: PASS (observed 1.0 fraction)
  behavior.trajectory_agreement@0.3.4 >= 0.99: PASS (observed 1.0 fraction)
  PASS means the user-defined criteria were met; it is not a universal deployment-safety judgment.
