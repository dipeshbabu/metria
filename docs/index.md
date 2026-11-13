---
title: Overview
hide:
  - navigation
  - toc
---

<div class="metria-kicker">Metria documentation</div>

<div class="metria-hero" markdown="1">
<div markdown="1">

# Test inference changes.<br><em>Keep the evidence.</em>

<p class="metria-lede">Compare a reference with a candidate, check what actually ran, and decide whether the change meets your acceptance criteria. Keep a report you can inspect and reproduce.</p>

<p class="metria-actions" markdown="1">
[Get started &rarr;](guides/getting-started.md){ .md-button .md-button--primary }
[See your first report](guides/first-report.md){ .md-button }
</p>

</div>
<div class="metria-terminal" markdown="1">
<div class="metria-terminal-title"><span></span><span></span><span></span><strong>A first look · no model required</strong></div>

```text
python -m pip install metria

metria demo --case pass --output demo-pass

# Read the synthetic decision and evidence
# demo-pass/report.md
# demo-pass/verification.json
```

<div class="metria-terminal-note">Synthetic example. Real verification uses a qualified runtime and pinned model.</div>
</div>
</div>

<div class="metria-facts"><span>Python 3.10–3.14</span><span>No base runtime dependencies</span><span>CLI + Python API</span><span>Apache-2.0</span></div>

## Start with a decision

Metria checks one declared inference change at a time. It keeps evidence gates,
behavioral measurements, and your acceptance policy separate, so a missing capture
or an unexpected difference cannot silently become a policy pass.

<div class="metria-cards" markdown="1">
<div class="metria-card" markdown="1">
<span class="metria-card-number">01 / LEARN THE WORKFLOW</span>

### Read your first report

Install the package and try synthetic PASS, FAIL, and NOT_COMPARABLE cases. Learn the output before setting up an inference engine.

[Run the walkthrough &rarr;](guides/first-report.md)

</div>
<div class="metria-card" markdown="1">
<span class="metria-card-number">02 / CHECK A REAL CHANGE</span>

### Verify a candidate

Choose a scoped workflow for local CPU settings, a runtime build, weight quantization, prefix caching, or serving measurements.

[Choose a workflow &rarr;](guides/verification-overview.md)

</div>
<div class="metria-card" markdown="1">
<span class="metria-card-number">03 / MAKE IT REPEATABLE</span>

### Gate changes in CI

Declare acceptance criteria, use distinct failure exits, and retain the report and both run records when a check fails.

[Set up CI &rarr;](guides/verification-ci.md)

</div>
</div>

## From change to evidence

<div class="metria-flow">
<div><strong>1. Declare</strong><span>Pin the model, runtime, workload, and intended change in a recipe.</span></div>
<div><strong>2. Verify</strong><span>Check observed identities and reject undeclared differences.</span></div>
<div><strong>3. Measure</strong><span>Inspect token drift, task checks, repeatability, and compatible timing.</span></div>
<div><strong>4. Decide</strong><span>Evaluate your policy and retain both runs with the report.</span></div>
</div>

[Understand the verification contract](architecture/verification-contract.md)
&nbsp; · &nbsp;
[Read the decision model](guides/verification-decisions.md)

<div class="metria-boundary" markdown="1">
**A valid comparison is the beginning of a decision.** `VERIFIED` means the
comparison had sufficient evidence and its analysis completed. A policy `PASS`
applies to your declared criteria on this workload. Neither is a universal model
quality or production performance guarantee. See [qualified and staged capabilities](guides/staged-capabilities.md).
</div>

## Go deeper

| Work on | Read next |
| --- | --- |
| Recipes and saved evidence | [Recipe CLI](guides/metria-recipe-cli.md), [study recipes](guides/metria-study-recipes.md), [run records](guides/metria-run-records.md) |
| Local inference engines | [llama.cpp adapter](guides/metria-llamacpp-runtime.md), [vLLM adapter](guides/metria-vllm-runtime.md), [runtime qualification](guides/runtime-qualification.md) |
| Fidelity measurements | [Unified fidelity tools](guides/unified-fidelity.md), [report comparison](guides/kv-fidelity-comparison.md), [token trajectories](guides/metria-trajectory-measurement.md) |
| Reproducible evidence | [Artifact resolution](guides/artifact-resolution.md), [testing evidence](guides/testing-evidence.md), [maintainer pilot](guides/representative-maintainer-pilot.md) |
| Development | [Core architecture](architecture/metria-core.md), [build these docs](maintainers/documentation.md), [contributing](../CONTRIBUTING.md) |

Current guidance lives here. [Historical research](../research/README.md) and
[retained artifacts](../artifacts/README.md) remain in the repository with their
original scope and provenance.
