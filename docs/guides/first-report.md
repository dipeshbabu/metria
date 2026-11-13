# Your first verification report

See the full result format before installing a model or inference engine.
This walkthrough uses bounded synthetic fixtures included in Metria.

## 1. Install Metria

Use a virtual environment with Python 3.10–3.14:

```bash
python -m venv .venv
```

=== "Linux / macOS"

    ```bash
    source .venv/bin/activate
    python -m pip install metria
    ```

=== "Windows PowerShell"

    ```powershell
    .venv\Scripts\Activate.ps1
    python -m pip install metria
    ```

```bash
metria --version
```

The base package has no runtime dependencies. Native inference engines are
installed separately when you move to a real comparison.

## 2. Produce a passing example

```bash
metria demo --case pass --output demo-pass
```

Open `demo-pass/report.md` in a Markdown viewer. The bundle also retains the
canonical `verification.json` and both run records. Choose a new output directory
for each invocation; Metria does not overwrite existing evidence.

!!! note "This result is synthetic"
    Every demo report labels its fixture evidence. A demo teaches the workflow;
    it does not qualify a runtime, demonstrate model quality, or measure real
    inference performance.

## 3. Read the report in order

| Section | What to inspect |
| --- | --- |
| **Change** | The intended difference between the reference and candidate. |
| **Evidence** | Whether execution completed and the observed comparison controls are valid. |
| **Impact** | Behavioral drift, configured task checks, reference repeatability, and compatible measurements. |
| **Verdict** | The comparison result and any explicit policy decision. |

Read execution and comparison status before interpreting measurements or policy.
Missing required evidence cannot become a passing acceptance decision.

## 4. See the failure cases

Run these commands separately. They intentionally return nonzero exits:

```bash
metria demo --case fail --output demo-fail
```

```bash
metria demo --case not-comparable --output demo-invalid
```

| Demo case | Expected exit | Meaning |
| --- | ---: | --- |
| `pass` | `0` | The synthetic comparison meets its declared policy. |
| `fail` | `1` | The comparison completes but the policy rejects the candidate. |
| `not-comparable` | `3` | An unexpected control difference prevents comparison. |

A policy rejection and an invalid comparison require different follow-up work.
Keep the report for either outcome rather than collapsing every nonzero exit
into a generic test failure.

## 5. Move to a real workload

[Choose a verification workflow](verification-overview.md), then follow its
runtime and model qualification instructions. Prepare a recipe, validate it, and
run the comparison:

```bash
metria recipe validate study.json
metria verify study.json --output verification --json
```

Declare your workload's [task checks and acceptance policy](verification-decisions.md).
The same recipe can run in [CI](verification-ci.md), where the report and run
records become reviewable artifacts.
