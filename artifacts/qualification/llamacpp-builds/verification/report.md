# Metria Verification

**Verdict: PASS**

Recipe: `27debdea0979aa2a1efcb137b55643173d156697c9aca47c00ca7975c11477d5`
Scope: local llama.cpp CPU build comparison

## Change:
  Capture provider SHA256: 5e17486285ea5ecf5bce1bfd7bafa172fe3e0b0fea0ede0eafa59fe5c998c9b6 -> 0fe6d5edf465771841c16dcdb52119a61c1a9d7300655f1ca8701dcd11141f42

## Evidence:
  reference: completed (reference.run.json)
    Observed threads: 2; context: 256
  candidate: completed (candidate.run.json)
    Observed threads: 2; context: 256
  Lifecycle: completed
  measurements: matched
  model: matched
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
  reference mean process wall time: 0.0151026s (includes startup and model loading)
  candidate mean process wall time: 0.0173709s (includes startup and model loading)
  Cold-process request latency: +0.00226829s (regressed)
    Relative change: +15.0192%
    One cold invocation per prompt; no statistical speedup claim.
  TTFT, decode throughput, inter-token latency, and device/KV memory: unavailable.

## Verdict:
  PASS
  Policy: PASS

Policy checks:
  quality.candidate_pass_rate@1 >= 1.0: PASS (observed 1.0 fraction)
  PASS means the user-defined criteria were met; it is not a universal deployment-safety judgment.
