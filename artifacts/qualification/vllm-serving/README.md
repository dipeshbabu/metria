# Native local serving qualification

Both pinned vLLM `0.30.0+cpu` and `0.30.0+cu129` runs passed their declared
policies. The CUDA run used the existing GTX 1650 (4 GiB, SM75) with automatic
cache dtype. This does not qualify FP8. The CPU run explicitly leaves CUDA
memory unavailable.

The original installed-wheel runs used source
`48ec730100903b969c8000333c24a6ef75129d7e` and wheel SHA256
`f2be41887bf7de15df54f86731a1c2ee1f9e459c32e5897d00f82b48083934dc`.
Each run retained one warmup and two measured batches, two authored public
prompts, native token captures, and nonempty task checks. Reference concurrency
was one and candidate concurrency two; engine capacity stayed at two, with
prefix caching disabled and public resets before every trial.

Each directory retains the original preparation/validation logs, recipe, policy,
workload, reference/candidate records, report and qualification receipt.
`current-projection.json` recomputes the report's systems fields from those exact
records using source `e63d3301eed3fdb9f2fe3c1704737bd9f69bf4b4`. That follow-up
corrected the inherited legacy `unsupported_metrics` summary; every measured
value, availability row, policy and verdict is unchanged. The original bundles
remain unmodified. An integration test verifies both their byte hashes and the
current evaluator's reproduction of the native evidence.

The [guide](../../../docs/guides/serving-measurements.md) defines measurement
boundaries. These are small-workload maintainer contract checks, not production
benchmarks or participant feedback. In particular, local stream timing excludes
HTTP/network overhead, and CUDA memory covers PyTorch allocator peaks and
allocated KV storage rather than total device use or live cache occupancy.
