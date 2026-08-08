# Metria runtime adapter qualification

Metria separates **semantic adapter conformance** from **real-engine/hardware
qualification**. Passing unit or mocked integration tests does not prove that a
runtime works on every upstream version, model architecture, kernel, or device.

## Qualification levels

| Level | What it proves | Where it runs |
|---|---|---|
| Shared semantic contract | Adapter obeys the Metria probe → resolve → launch → infer → observe → reset → close lifecycle, keeps evidence privacy boundaries, resolves deterministically, and fails after close | PR-required Metria core tests |
| Engine-specific contract tests | Runtime-specific flags, token capture, launch arguments, redaction, cleanup, and configured-vs-observed semantics behave as implemented | PR-required Metria core tests |
| External-boundary adapter conformance | The real Metria adapter class satisfies the shared semantic contract while only the external engine process/module is replaced by a deterministic fake | PR-required Metria core tests |
| Real-engine qualification | A pinned upstream runtime actually launches and produces expected observed identity/capture semantics | Manual or scheduled environment with the runtime installed |
| Hardware-qualified evidence | A real runtime/model executes on identified accelerator hardware and retains the model/runtime/hardware evidence needed to reproduce the qualification | Manual or scheduled hardware runner |

The first three levels are normal pull-request gates. The last two require an
explicit environment and should not be inferred from mocked tests.

## Current first-party adapters

### llama.cpp

PR-required coverage includes:

- shared runtime semantic contract exercised through `LlamaCppAdapter`;
- binary/model existence and content hashes during resolution;
- command construction and managed-flag rules;
- timeout/non-zero exit behavior;
- prompt/system redaction in retained invocation evidence;
- KV-cache runtime-feature handling;
- optional token-ID capture when a compatible completion binary is present;
- reset/close semantics.

Real-engine qualification still needs a pinned llama.cpp build, a model artifact
with immutable identity, and hardware evidence. The default CI suite does not
make that claim.

### vLLM

PR-required coverage includes:

- shared runtime semantic contract exercised through `VLLMAdapter`;
- lazy optional dependency behavior;
- constructor/runtime configuration;
- batched generation and token-ID capture;
- prompt/system redaction;
- configured-vs-introspected applied state;
- reset/close semantics.

Real-engine qualification still needs a pinned vLLM environment, immutable model
identity, and accelerator evidence. The default CI suite does not make that
claim.

## Evidence required for a real qualification

A real-engine qualification should retain at least:

```text
Metria version / commit
runtime name + upstream version/commit
model identifier + immutable revision/digest
tokenizer identifier + immutable revision/digest
requested runtime configuration
resolved configuration
observed/applied runtime evidence
hardware fingerprint + accelerator identity
driver/runtime software versions
measurement/capture method + version
run status and lifecycle events
```

Where an engine cannot expose authoritative applied state, the qualification
must say `unknown`/`unverified` rather than copying requested values into the
observed record.

## CI policy

- Mocked/fixture conformance remains required for every PR because it is
  deterministic and cross-platform.
- Real-engine/hardware qualification is **not** a required GitHub-hosted PR gate
  until Metria has a controlled runner, immutable model artifacts, and pinned
  upstream runtime inputs.
- When a scheduled/manual qualification lane is added, its artifacts should be
  versioned Metria run records/manifests rather than ad-hoc console logs.
- A failed or stale real-engine qualification should downgrade published support
  claims; it must not be hidden by passing mocked tests.

## Future runtimes

MLX, SGLang, TensorRT-LLM, or other engines should not be advertised as
first-party Metria runtime support merely because a focused component has a
backend for them. A new Metria runtime should first implement the common adapter
protocol and pass the same semantic/conformance levels described here.
