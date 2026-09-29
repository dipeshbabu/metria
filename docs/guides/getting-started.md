# Test an inference change

Metria compares one reference configuration with one candidate. It checks the
requested change actually applied, rejects unexpected differences, measures
behavioral drift, and reports whether your explicit acceptance criteria passed.

Start with the [copyable examples](../../examples/verification/README.md).
The synthetic cases demonstrate PASS, FAIL, and NOT_COMPARABLE without a GPU or
model download. They are labeled test evidence and do not qualify a real runtime.

Install the unified package with Python 3.10–3.14:

```bash
python -m pip install metria==0.1.2
```

For a real comparison, follow the [local llama.cpp guide](metria-verify.md) to
prepare the pinned model, qualified capture provider, and CPU thread-change recipe:

```bash
metria verify study.json --output verification --json
```

Read `report.md` in this order: **Change → Evidence → Impact → Verdict**.
`verification.json` retains the same canonical result for automation. If evidence
is missing or an undeclared control differs, comparison stops before a policy can
approve the candidate. The report identifies the missing evidence or difference.

Add [your acceptance policy](verification-policies.md) when a valid comparison
must meet numeric or exact criteria. Metria does not choose universal behavior or
safety thresholds. Use the [CI guide](verification-ci.md) to run the same check in
code review and retain artifacts on failure.

Metria 0.1.2 also supports scoped build/quantization comparisons,
[vLLM prefix-cache verification](vllm-prefix-verification.md), isolated CPU
runtime upgrades, and [local serving measurements](serving-measurements.md).
The [release notes](../releases/0.1.2.md) list qualified boundaries and deferred
FP8 work. The same package includes [unified fidelity tools](unified-fidelity.md)
and [pilot decision records](verifier-pilot.md).

Recipe validation, inspection, and saved-record comparison are supporting tools
for this workflow. Find them in the [documentation index](../index.md).

The former experimental-engine setup remains in the
[historical TurboQuant+ archive](../../research/archive/turboquant-engine-setup.md).
It requires historical forks and does not describe the current verifier.
