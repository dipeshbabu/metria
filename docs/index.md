---
title: Introduction to Metria
---

# Introduction to Metria

<p class="metria-intro">Metria verifies changes to LLM inference systems. Compare a reference with a candidate, check what actually ran, measure behavioral change, and evaluate the result against your acceptance criteria.</p>

It retains both runs and a report you can review locally or in CI. Execution
evidence, comparison validity, and policy acceptance remain separate: missing
captures or unexpected configuration differences cannot silently become a pass.

## Getting started

Install Metria in a virtual environment with Python 3.10–3.14:

```bash
python -m pip install metria
```

The base package has no runtime dependencies. Native inference engines and
models are installed separately for real verification. You can see the complete
result format immediately with a synthetic example:

```bash
metria demo --case pass --output demo-pass
```

Open `demo-pass/report.md` to read the decision. The same bundle contains the
canonical `verification.json` and both run records.

!!! note "Start without a model or GPU"
    The demo is explicitly synthetic. It teaches the report format and failure
    states; it does not qualify a runtime, demonstrate model quality, or measure
    real inference performance.

<p class="metria-actions" markdown="1">
[Your first report &rarr;](guides/first-report.md){ .md-button .md-button--primary }
[Installation and setup](guides/installed-workflow.md){ .md-button }
</p>

## What you can verify

Start with one declared change and a fixed workload. Each workflow states its
runtime, hardware, and evidence requirements.

<div class="metria-cards" markdown="1">
<div class="metria-card" markdown="1">
<img src="assets/icons/compare.svg" alt="" width="28" height="28">

### Inference changes

Compare CPU settings, runtime builds, controlled weight quantization, vLLM prefix caching, runtime upgrades, FP8 KV caches, or local serving measurements.

[Choose a workflow &rarr;](guides/verification-overview.md)

</div>
<div class="metria-card" markdown="1">
<img src="assets/icons/shield.svg" alt="" width="28" height="28">

### Acceptance criteria

Keep token drift, task-check outcomes, reference repeatability, and compatible timing separate. Define your own policy.

[Define a decision &rarr;](guides/verification-decisions.md)

</div>
<div class="metria-card" markdown="1">
<img src="assets/icons/terminal.svg" alt="" width="28" height="28">

### Repeatable checks

Run the same recipe in CI, interpret distinct exit codes, and retain the evidence even when a comparison fails.

[Set up CI &rarr;](guides/verification-ci.md)

</div>
</div>

## How verification works

1. **Declare the change.** Pin the model, runtime, workload, and intended variation
   in a [study recipe](guides/metria-study-recipes.md).
2. **Check the evidence.** Verify observed identities and reject undeclared
   differences before treating the runs as comparable.
3. **Measure impact.** Inspect token trajectories, configured task checks,
   repeatability, and compatible systems measurements.
4. **Evaluate your policy.** Apply explicit criteria and retain the result with
   both run records.

`VERIFIED` means the comparison had sufficient evidence and its analysis completed.
A policy `PASS` applies to your declared criteria on this workload. Neither is a
universal model-quality or production-performance guarantee.

Read the [verification contract](architecture/verification-contract.md) and
[qualified and staged capability notes](guides/staged-capabilities.md) for the
precise boundaries.

## Read next

| Work on | Documentation |
| --- | --- |
| Run a real comparison | [Choose a workflow](guides/verification-overview.md), [installed workflow](guides/installed-workflow.md) |
| Build with the CLI and Python API | [Recipe CLI](guides/metria-recipe-cli.md), [run records](guides/metria-run-records.md), [pairwise analyses](guides/metria-pairwise-analysis.md) |
| Inspect behavior | [Fidelity tools](guides/unified-fidelity.md), [token trajectories](guides/metria-trajectory-measurement.md) |
| Understand retained evidence | [Artifact resolution](guides/artifact-resolution.md), [testing evidence](guides/testing-evidence.md) |
| Contribute | [Core architecture](architecture/metria-core.md), [build these docs](maintainers/documentation.md), [contributing](../CONTRIBUTING.md) |

[Historical research](../research/README.md) and
[retained artifacts](../artifacts/README.md) remain in the repository with their
original scope and provenance.
