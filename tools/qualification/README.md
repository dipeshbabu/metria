# Local llama.cpp capture provider

The first local verifier uses a pinned CPU build of llama.cpp with a small
downstream capture patch. The patch writes sampled token IDs to the existing
`KV_FIDELITY_TRAJECTORY` JSONL stream, plus a `.runtime.json` sidecar containing
context size, active thread counts, vocabulary size, and whether chat formatting
was applied. It does not change sampling or write prompt text to those files.

Upstream source: `https://github.com/ggml-org/llama.cpp`

Revision: `434ddbbc0e30522e897670681e503b797c12b7c1`

Apply `llamacpp-capture.patch` to that revision and build `llama-completion` with
CMake. Metria accepts a completion-only binary directory. The downstream patch
is maintained here; it is not an upstream llama.cpp feature or support claim.

The token provider must be qualified and its SHA-256 supplied through
`environment.llama_cpp_token_ids_capture_sha256`. An unpatched provider remains
insufficient for the local verification workflow. Qualification is specific to
the recorded build, model, host, and workload, not a claim about other engines
or hardware.

The patch is derived from MIT-licensed llama.cpp; see `LICENSE.llama.cpp`.

`staged/llamacpp-topk-capture.patch` is an unqualified extension applied after
`llamacpp-capture.patch`. When `KV_FIDELITY_TOPK` is set it adds full-vocabulary
top-k log probabilities to each trajectory row; otherwise output is unchanged.
It stays separate so qualified providers keep the SHA-256 of the capture patch.
After building it on top of a qualified source tree with the same CMake flags,
`staged/check_topk_provider.py` checks the layered provider against the
qualified one. See [staged capabilities](../../docs/guides/staged-capabilities.md).

## Staged vLLM GPU observations

`staged/setup_vllm_cuda.sh NEW_WORK_DIR` prepares the pinned `0.30.0+cu129`
environment on a Linux GPU host: it verifies the wheel hash, installs the
recorded torch/setuptools pins, and downloads only the descriptor's model files.
Some GPU images set `UV_NO_CACHE=1` and `UV_LINK_MODE=copy`, which stores every
package twice; the script keeps one symlinked copy so the stack fits on a 16 GB
disk. `staged/observe_vllm_gpu.py run --variant reference|fp8|staged` then
launches one engine per process through Metria's adapter, and `compare` derives
trajectory agreement from the saved records. The retained
[L4 observations](../../artifacts/qualification/l4-native-gpu/README.md) were
produced this way.

## Linux toolchain

The build scripts need Linux (or WSL2) with `git`, CMake 3.28 or newer, GNU
Make and a C/C++ compiler; Ubuntu 24.04's `build-essential` and `cmake` packages
are sufficient. Build from a Linux checkout of this repository: `git apply`
rejects the patches if a Windows checkout converted them to CRLF line endings.
A rebuilt provider is not bit-for-bit identical to a retained one, so qualify it
by its own SHA-256 before using it in a qualified profile.

`build_llamacpp_cpu.sh` builds that pinned provider in a new workspace.
`prepare_cpu_verification.py` exercises it, verifies the local model pin, saves
the qualification run, and creates a one-thread versus two-thread recipe.
See the [complete walkthrough](../../docs/guides/metria-verify.md).
