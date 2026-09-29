# Validate usefulness on a user's workload

Automated checks and maintainer-run hardware qualification establish implementation
evidence. They do not establish external adoption or that the workflow fits every
inference team's decisions. External pilot feedback remains to be collected from
real participants. Maintainer validation and participant feedback are recorded
separately.

## Pilot task

Use an already installed supported runtime and model. Bring a small representative
workload, a trusted model/tokenizer manifest and explicit acceptance criteria.
Prepare and verify one supported reference/candidate change using only the installed
Metria commands. Keep the recipe and report; share records only after reviewing
paths, token IDs and workload information for privacy.

Record the result before asking a maintainer for help. Include failed setup and
negative results rather than selecting only successful runs.

## Save a decision using the installed package

After `metria verify`, write `pilot-notes.json` with your observations:

```json
{
  "kind": "maintainer_validation",
  "reporter": "your identifier",
  "decision": "inconclusive",
  "rationale": "Describe what the report establishes and what remains uncertain.",
  "acceptance_criteria": "Describe the task checks and performance thresholds declared before running.",
  "setup_seconds": 120,
  "friction": ["Describe a concrete setup problem, or use an empty array."],
  "would_reuse": null,
  "permission_to_share": false
}
```

Then run:

```bash
metria pilot record --evidence verification \
  --notes pilot-notes.json --output pilot-record.json
```

The command checks both run records against the saved verification, checks the
report projection and retains hashes of the exact files it read. It records your
decision alongside the verification verdict; it does not infer satisfaction or
independently rerun the experiment. Both failed and successful verification bundles
can support a record. The command refuses existing output and performs no upload.

For actual participant feedback, use `"kind": "participant_feedback"` and add
`"feedback_source"` naming the retained response or authorized feedback reference.
Keep `permission_to_share` false unless that participant explicitly permits
sharing. A source reference records supplied provenance; it does not authenticate
the participant. Automated maintainer runs must keep `would_reuse` null and must
not be presented as another person's feedback.

The table below remains available for participants who prefer free-form notes.

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
