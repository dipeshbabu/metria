# Separate drift, task checks and systems impact

Token disagreement detects a behavioral change. It does not establish that an
answer became worse. The Metria 0.1.2 vLLM workload can retain deterministic task checks,
reference-repeatability evidence and compatible latency alongside trajectories.
`metria.verification_impact` version `1` keeps these results separate.

## Declare checks in the workload

Each JSONL prompt can include a `checks` array:

```json
{"id":"shelves","prompt":"The archive has twelve shelves. Reply with the number of shelves only.","checks":[{"id":"correct-count","kind":"exact_text","expected":"12","normalization":"strip"}]}
```

| Check | Meaning |
|---|---|
| `exact_text` | Equality with declared expected text; normalization is `strip` (default) or `none`. |
| `nonempty` | The output contains a non-whitespace character. This is a basic response check, not a semantic-quality assessment. |
| `json_object` | The output is a JSON object containing every declared `required_keys` entry. Duplicate keys and nonfinite constants fail. |

Checks run after each measured inference call, outside the latency clock. Warmup
results do not count. A pass rate describes only the checks and prompts that the
user declared; the report shows the checked fraction of the workload. No checks
means unavailable quality evidence, not a zero score or an automatic pass.

The recipe contains expected answers and should be treated as sensitive input.
Summary evidence retains check definitions and output identities by digest, plus
IDs, kinds and outcomes. It does not add raw answers to summaries. Run records
still contain token IDs and requested configuration; those can reveal response
content when decoded and must be reviewed before sharing.

## Choose an explicit policy

Recipes prepared by the current vLLM preparation API request both trajectory and
verification-impact analyses. A policy may combine their independently versioned
targets:

```json
{
  "schema": "metria.verification_policy.v1",
  "criteria": [
    {"target":"quality.candidate_pass_rate","version":"1","min":1.0},
    {"target":"quality.pass_rate_delta","version":"1","min":0.0},
    {"target":"performance.latency_ratio","version":"1","max":1.1},
    {"target":"behavior.reference_repeatability","version":"1","min":1.0}
  ]
}
```

These are illustrative user choices, not shipped acceptance defaults. A latency
ratio of 1.1 allows a candidate mean up to 10% above the reference under the same
complete measurement methodology. Zero or near-resolution baselines cannot yield
a ratio and remain unavailable.

| Impact target, version `1` | Unit/domain |
|---|---|
| `quality.candidate_pass_rate`, `quality.reference_pass_rate` | Fraction, 0–1 |
| `quality.pass_rate_delta` | Candidate minus reference fraction, −1–1 |
| `behavior.reference_repeatability` | Fraction of equal token sequences across observed reference repeat pairs, 0–1 |
| `performance.candidate_latency_seconds` | Nonnegative seconds |
| `performance.latency_ratio` | Nonnegative candidate/reference ratio |

Existing trajectory targets retain their `0.3.4` versions and meanings. Declared
policy targets require their matching analysis; missing/incompatible metrics or
incomplete coverage produce `INSUFFICIENT_EVIDENCE`, even for a permissive threshold.
Execution, evidence and comparison failures prevent policy evaluation entirely.

## Interpret repeated evidence

Reference repeatability compares complete repeated token sequences for each prompt.
The report states how many observed repeat pairs differed and how many prompts and
trials were available. One trial cannot establish repeatability. This sample result
is not a confidence interval, a guarantee of determinism or a quality score.

The report places task-check counts, failed check IDs, reference variation and
latency limitations beside behavioral divergence. `PASS` means the specified policy
was met on this workload and evidence. It is not a universal deployment judgment.
Legacy expert composite bands remain diagnostic conventions and do not replace
these explicit acceptance criteria.
