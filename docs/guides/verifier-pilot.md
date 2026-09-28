# Validate usefulness on a user's workload

Automated checks and maintainer-run hardware qualification establish implementation
evidence. They do not establish external adoption or that the workflow fits every
inference team's decisions. External pilot feedback remains to be collected from
real participants; this document contains a blank protocol, not claimed results.

## Pilot task

Use an already installed supported runtime and model. Bring a small representative
workload, a trusted model/tokenizer manifest and explicit acceptance criteria.
Prepare and verify one supported reference/candidate change using only the installed
Metria commands. Keep the recipe and report; share records only after reviewing
paths, token IDs and workload information for privacy.

Record the result before asking a maintainer for help. Include failed setup and
negative results rather than selecting only successful runs.

| Field | Participant's observation |
|---|---|
| Participant/team and date | |
| Metria version or immutable source revision | |
| Runtime version, hardware, model/tokenizer pins | |
| Intended inference change | |
| Workload and task checks, with private data omitted | |
| Time from installed prerequisites to first valid report | |
| Commands or configuration that required assistance | |
| Could you explain why the comparison was valid or rejected? | |
| Did task checks and timing help your keep/reject decision? | |
| Decision and supporting criteria | |
| Most confusing output or missing capability | |
| Would you run this on the next relevant infrastructure change? | |
| Evidence link and sharing permission | |

## Completion evidence

Keep automated wheel/sdist/CLI tests, maintainer CPU/CUDA runs and external feedback
as separate records. For each external pilot, retain the participant's actual
feedback and report reference with permission. Do not infer satisfaction from CI
passing or use synthetic fixtures as pilot evidence.

Prioritize recurring setup friction, invalid-comparison explanations and decisions
that the report could not support. Expand scope only after the current workflow
has demonstrated value on those workloads.
