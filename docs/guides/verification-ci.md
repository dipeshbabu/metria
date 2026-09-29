# Gate inference changes in CI

Run the verifier directly and preserve its exit status:

```bash
metria verify .metria/change.json --output metria-verification --json
```

Prepare the pinned model, qualified runtime and recipe before this step. Use a
controlled runner for real engine evidence. The current CLI supports the local
llama.cpp CPU thread workflow described in [the verification guide](metria-verify.md).

## Stable exit codes

Metria distinguishes these outcomes:

| Exit | Outcome |
|---|---|
| 0 | `VERIFIED`, or explicit user policy `PASS` |
| 1 | User policy `FAIL` |
| 2 | Invalid recipe/configuration, including an unreadable recipe |
| 3 | `NOT_COMPARABLE` |
| 4 | `INSUFFICIENT_EVIDENCE` |
| 5 | `EXECUTION_FAILED`, including preflight or output persistence failure |
| 130 | Interrupted verification |

These replace the development command's previous catch-all unsuccessful exit 1.
Shell gates that treat every nonzero code as failure continue to work. Integrations
that matched only exit 1 must handle all nonzero codes. Published 0.1.0 retains
its original exits and does not contain the newer CI/reporting features.

Completed bundles expose `verdict`, `exit_code`, `lifecycle`, `comparison_status`,
and `policy_status` in `verification.json`. Invalid inputs or persistence failures
emit `metria.verification_error.v1` on JSON stdout with an error type and message
hash, without copying exception values or secrets. A missing canonical file means
there is no complete bundle; inspect retained records and the command status.

## GitHub Actions

[Copyable workflow](../../examples/verification/github-actions.yml) installs the
checked-out Metria revision, runs verification, appends a report to the job summary,
and uploads the evidence directory even when the verifier fails. It intentionally
uses neither `continue-on-error` nor shell constructs that replace a failed exit
status. Add your controlled runtime/model provisioning before verification and
enable a pull-request trigger only for an appropriate runner and workload.

For an external repository, pin `metria==0.2.0` or an immutable Metria source
revision and copy the small summary adapter with the workflow. Use the matching
recipe profile and explicit runtime/model pins on the controlled runner.

`tools/ci/write_verification_summary.py` is a thin provider adapter around canonical
JSON. Core verification does not inspect GitHub environment variables. Other CI
providers can retain the same result and Markdown files and use the same exits.
Reports omit prompts and generated text; inspect raw run-record configuration
before uploading it. Use non-sensitive prompt identifiers and category names.

The required core test matrix exercises the full command with a deterministic
fake llama.cpp subprocess, including PASS, policy failure, invalid comparison,
insufficient capture, runtime failure, and interruption. Summary integration tests
use that same fixture and require no model download or accelerator.
