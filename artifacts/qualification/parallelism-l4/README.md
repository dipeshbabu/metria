# Native 2×L4 qualification of vLLM tensor/pipeline parallelism

On 2026-11-23 `local_vllm_parallelism.v1` ran end to end on a vast.ai host with
two NVIDIA L4 GPUs (compute capability 8.9, driver 595.91.07, connected through
a PCIe host bridge, `PHB` in [`topology.txt`](topology.txt)), using the pinned
`vllm==0.30.0+cu129` wheel with `torch==2.13.0+cu129` and NCCL 2.29.7
([`packages.txt`](packages.txt)). The model is Qwen2.5-0.5B-Instruct at revision
`7ae557604adf67be50417f59c2c2f167def9a775` (14 attention heads, 2 KV heads),
pinned file by file in
[`tools/qualification/vllm-qwen2.5-0.5b-instruct.json`](../../../tools/qualification/vllm-qwen2.5-0.5b-instruct.json)
and verified before each run. The workload is four prompts with a `nonempty`
check, 16 greedy tokens, one warmup and two measured trials, with staged top-20
distributions. Both runs record both visible GPUs (hashed UUIDs) under
`observed.hardware.devices`.

| Case | Reference | Candidate (engine readback) | Verdict | Agreement | Trajectory KLD (nats) | Pass-rate delta |
| --- | --- | --- | --- | --- | --- | --- |
| `tp2` | 1 GPU | tensor parallel **2** | **FAIL** | 0.891 | 0.0077 | 0 |
| `pp2` | 1 GPU | pipeline parallel **2** | **PASS** | 1.0 | 0 | 0 |

Policy ([`policy.json`](policy.json)): pass-rate delta ≥ 0, trajectory
agreement ≥ 0.9, trajectory KLD ≤ 0.05 nats. Both comparisons are `VALID` with
no evidence gaps; only the declared parallel paths differed.

**What this shows.**

- Tensor parallelism splits each matrix multiply across GPUs and changes the
  order of floating-point reductions. The next-token distributions barely moved
  (mean KLD 0.0077 nats), yet greedy decoding diverged on some prompts and
  agreement fell just below the 0.9 bound, so the declared policy failed. This
  is the drift the profile exists to measure; a lower agreement bound or a
  KLD-only policy would accept it.
- Pipeline parallelism places whole layers on each GPU without splitting their
  arithmetic, and produced identical tokens and distributions.

**NCCL setting.** On this host the default NCCL peer-to-peer transport hung
during tensor-parallel initialisation (an earlier attempt timed out in
`pynccl`). Both cases ran with `NCCL_P2P_DISABLE=1`, which routes inter-GPU
traffic through host memory. The variable is inherited by the verification
worker; it is not recorded in the run records, so it is stated here.
[`tp_probe.log`](tp_probe.log) is the direct vLLM probe with that setting.

**A bug this run found.** The first attempt ended `EXECUTION_FAILED` because the
isolated verification worker did not yet accept the parallelism scope. The fix
is included in the same change as this evidence; the runs here used it
([`metria_head`](metria_head) is the base commit).

## Files and redaction

- `tp2/`, `pp2/`: the verification bundles (`verification.json`, `report.md`,
  both run records), plus the prepared `*.study.json`, logs, `workload.jsonl`
  and `policy.json`.
- The run records, recipes and logs named the working directory: the folder
  `mg` inside the root user's home directory on the GPU host. Exactly that
  absolute prefix is replaced with `${WORK}`;
  [`originals-sha256.json`](originals-sha256.json) lists the SHA-256 of every
  original file. Substituting the prefix back restores the original bytes,
  whose `run_record_digest` matches `verification.json`.
- [`files-sha256.json`](files-sha256.json) hashes every retained file.
