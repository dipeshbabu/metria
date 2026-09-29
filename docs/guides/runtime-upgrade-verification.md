# Verify a local vLLM CPU runtime upgrade

The Metria 0.1.2 workflow compares separately pinned vLLM
`0.29.0+cpu` and `0.30.0+cpu` environments on the same Linux/WSL machine. Model and
tokenizer files, generation settings, CPU placement and workload stay fixed.
Runtime installation contents and dependency versions are recorded independently.
This is a comparison of the installed runtime stacks; it does not attribute every
observed change to vLLM source alone.

Install the same Metria wheel in the controller and both runtime environments.
The two environments must use the same Python interpreter contents and separate
prefixes. Their vLLM dependencies remain independent. Use existing qualified
CPU wheels and retain their installation constraints and provenance.

```bash
metria recipe prepare-vllm-upgrade \
  --reference-python /path/to/vllm-0.29/venv/bin/python \
  --candidate-python /path/to/vllm-0.30/venv/bin/python \
  --model /path/to/pinned/local/model \
  --descriptor trusted-model-files.json \
  --workload prompts.jsonl \
  --policy acceptance-policy.json \
  --output upgrade-study.json

metria verify upgrade-study.json --output upgrade-verification
```

Preparation records each interpreter, prefix, Metria implementation, installed
vLLM payload/dependency fingerprint, and optional preloaded OpenMP library pin.
The selected CPU IDs must be available in both environments. It preserves the
virtual-environment interpreter path even when the executable is a symlink.
It does not install packages or download model files.

Each interpreter is restricted to the selected CPU set before engine startup,
so engine workers and threads inherit that placement. A named Metria worker
extension reads the union of native worker thread affinities before and after
the workload. No callable RPC serialization fallback is enabled. Environment
descriptors retain the interpreter's observed availability before this explicit
binding; the run record retains the native worker placement after it is applied.

Verification runs the normal Metria execution lifecycle once in each pinned
interpreter, one after the other. Each worker checks its environment before and
after execution. The controller checks request and run-record bindings before
accepting returned evidence and saves the reference before starting the candidate.
Changed installations, mismatched worker results and timeouts cannot produce
policy acceptance. A shared deadline bounds both runs and process-tree cleanup.

The qualified profile uses float32, one worker, eager greedy plain completion,
prefix caching disabled in both runs, and explicit cache resets between trials.
Native worker observations must confirm the controlled CPU placement. GPU runtime
upgrades, arbitrary version pairs and unrelated setting changes are separate
qualification scopes.

Task checks and acceptance policies have the same semantics as other verifier
profiles. Compatible loaded-engine request latency excludes engine startup,
warmup and cache reset. It does not become TTFT or serving throughput merely
because two runtime versions are being compared. Retain both run records and
interpret missing measurements as unavailable.

[Retained installed-wheel qualification](../../artifacts/qualification/vllm-cpu-upgrade/README.md)
includes the two pinned native environments, actual worker placement, both run
records and the policy result. The first qualified version pair uses the tiny
local model as contract evidence; broader version/device pairs need their own
qualification.
