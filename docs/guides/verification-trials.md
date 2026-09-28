# Repeated verification for research utilities

`execute_verification_trials()` repeats the existing qualified verifier with an
explicit warmup/measured-pair policy. Each pair uses fresh runtime sessions and
retains its canonical report and run records. The current llama.cpp scope starts
a fresh completion process per prompt, with the existing bounded timeout and
process-tree cleanup. No independent inference service or fixed readiness sleep
is introduced.

```python
from metria import load_study_recipe
from metria.verification_trials import VerificationTrialPolicy, execute_verification_trials

result = execute_verification_trials(
    load_study_recipe('study.json'), 'trial-results',
    policy=VerificationTrialPolicy(warmup_pairs=1, measured_pairs=3),
)
```

Warmups repeat full invocations and may warm filesystem/startup caches; they do
not preserve a loaded engine. Their evidence is retained but excluded from the
measured latency samples. Runtime readiness, model/provider verification, applied
configuration checks, workload identity, comparison, and cleanup all use the
existing verifier. A failed evidence/comparison/execution gate stops further pairs.
Policy outcomes belong to the explicit recipe; there are no model/hardware-specific
default thresholds.

The summary records every pair, trial policy, native cold-request timing samples
when available, and an identity key covering model/provider content, recipe/workload,
hardware evidence, framework source, methods, and trial policy. A cached baseline
is eligible only when that identity matches. This key is an eligibility check,
not a universal comparability proof or an automatic performance policy. TTFT,
decode-only throughput, and device/KV memory are not inferred from wall time or
word counts. Missing timing stays unavailable.

## Migration from the historical benchmark scripts

The old quick/real-world scripts mixed unpinned service startup, fixed sleeps,
path interpolation into generated Python, model-specific reference numbers, and
PDF fallbacks. They are now thin compatibility entry points for the qualified
verification-trial API:

```bash
bash tools/benchmarks/turbo-quick-bench.sh \
  --recipe study.json --output trial-results --measured-pairs 3
```

`turbo-realworld-bench.sh` accepts the same arguments. Legacy positional model/PDF
arguments must be migrated to an explicit prepared recipe; they are no longer
silently interpreted. Paths with spaces or quotes are passed as arguments/data.
Prepare any private document workload explicitly and retain its identity in the
recipe instead of interpolating its text into source. `--no-ref` remains a
compatibility option for omitting a cached baseline; it does not disable a recipe
policy. `--baseline` accepts a prior `verification-trials.json`.

This is a research utility around one reference/candidate workflow. It does not
add `metria benchmark`, generic sweeps, a server framework, or an arbitrary plugin
surface. Historical scripts remain recoverable from Git history for interpreting
dated results; their old output is not relabeled as current qualified evidence.

The vLLM prefix-cache profile retains its own loaded-engine timing boundary and
internal warmup/cache-reset policy. Outer pair repetitions do not relabel that
latency as cold-process timing or preserve engine state across pairs.
