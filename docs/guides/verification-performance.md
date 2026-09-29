# Performance impact within verification

The verifier reports a method-identified performance delta only
after the reference/candidate comparison and required evidence gates pass.
The first measurement is **cold-process request latency** for local llama.cpp:
the adapter's monotonic duration includes process startup, model loading, prompt
evaluation, and generation. It describes the verification workload's complete
invocation cost.

`metria.llamacpp_process_request_latency@1` records seconds, lower-is-better
direction, zero warmups, one trial per prompt, a fresh process per prompt,
arithmetic mean aggregation, sample count, and a workload fingerprint. Raw clocks
remain in the referenced run records. Prompt fingerprints must match their
corresponding successful runtime invocations; token evidence must be present.

The canonical result's `performance` field includes reference/candidate means,
absolute change, relative change, direction, and methodology. Relative change is
`(candidate - reference) / reference`; a negative latency delta is an improvement.
Baselines at or below one nanosecond and non-finite ratios produce an explicit
unavailable relative delta. They never produce an infinite speedup.

Units, method/version, aggregation, trial policy, workload identity, and full
coverage must agree. Missing, malformed, partial, method-incompatible, or
non-comparable evidence produces an unavailable result with its reason. No
performance benefit is inferred from a failed comparison.

TTFT, inter-token latency, decode-only throughput, peak device memory, and KV
memory are explicitly unavailable in this path. Process wall time is not a
substitute for those measurements, and output word counts are never used as token
counts. Repeated-trial policy is fixed at one trial per prompt in this initial
scope; the report makes no statistical speedup claim. Runtime-native timing or
memory capture requires a separately qualified method before those metrics can
be added.

Python callers can use `measure_invocation_performance()` and
`compare_performance()` from `metria.measurements.performance` with existing
`RunRecord`/`MeasurementResult` objects. These functions reuse retained runtime
evidence and do not schedule another inference workload.

The Metria 0.1.2 [vLLM prefix-cache profile](vllm-prefix-verification.md) uses the distinct
`metria.runtime_call_latency` method after engine loading, with explicit cache
isolation, warmup and repetitions. Its values are not method-compatible with the
llama.cpp cold-process metric described above.

The [local serving profile](serving-measurements.md) uses separate streaming,
batch-throughput and native-memory methods. Its `performance.metrics` map retains
each metric's availability, units, method/version, reference/candidate values and
ratio. `serving.*` policy targets apply to these observations. Client-observed
token deliveries do not imply HTTP endpoint timing, and PyTorch allocation peaks
do not imply device-wide memory usage.
