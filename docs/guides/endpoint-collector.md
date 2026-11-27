# Host evidence collectors for endpoint verification

A remote inference server can only *claim* what it runs. A **collector** is a
small process you run beside each server you operate. When the verifier
challenges it, it reads facts the server cannot be trusted to self-report and
returns them signed and bound to that challenge. With collectors configured,
[remote endpoint verification](endpoint-verification.md) upgrades identity
from `claimed` to `collector_observed` for the facts it can read.

## What a collector reads

| Fact | Source |
| --- | --- |
| `engine.dtype`, `engine.kv_cache_dtype`, `engine.tensor_parallel_size`, `engine.pipeline_parallel_size`, `engine.max_model_len`, `engine.quantization`, `engine.enable_prefix_caching`, … | The vLLM server process's own command line (`/proc/<pid>/cmdline`). Omitted flags take vLLM's defaults (`auto`, 1, 1), listed in `engine.defaulted`. |
| `model.files` | SHA-256 of every model and tokenizer file in the served model directory |
| `gpu.devices`, `gpu.driver_version` | `nvidia-smi` (UUIDs hashed) |
| `environment.numerics` | The server process's `NCCL_*`, `VLLM_*`, CUDA and thread variables, credentials excluded, paths hashed |
| `image.digest` | Supplied by the operator (`--image-digest`) |

Anything that cannot be read is listed as unavailable, never guessed. Raw paths,
credentials and prompts never leave the host.

## Set up

On each serving host:

```bash
export METRIA_COLLECTOR_KEY=$(metria collector keygen)
metria collector serve --pid "$(pgrep -f 'vllm serve' | head -1)" \
  --key-id hosts-2026-11 --port 9105
```

The collector listens on `127.0.0.1` by default; expose it to the verifier the
same way you expose the inference endpoint (for example an SSH tunnel). Give the
verifier the key ring as JSON in an environment variable:

```bash
export METRIA_COLLECTOR_KEYS='{"hosts-2026-11": {"algorithm": "hmac-sha256", "key": "<base64>"}}'
export METRIA_REFERENCE_COLLECTOR=http://127.0.0.1:9105
export METRIA_CANDIDATE_COLLECTOR=http://127.0.0.1:9106
```

Then add collector options when preparing the recipe:

```bash
metria recipe prepare-endpoint … \
  --reference-collector-url-env METRIA_REFERENCE_COLLECTOR \
  --candidate-collector-url-env METRIA_CANDIDATE_COLLECTOR \
  --collector-keys-env METRIA_COLLECTOR_KEYS \
  --reference-deployment auto-pool --candidate-deployment fp8-pool \
  --required-fact engine.kv_cache_dtype
```

Key rotation: add a new key ID to the key ring, switch the collectors to it,
then remove the old ID.

## What the verifier enforces

Each run challenges its collector at launch and after the last request, each
time with a fresh nonce bound to the run's session, role and deployment.

| Condition | Outcome |
| --- | --- |
| Missing collector, unreachable collector, unknown key ID, invalid signature, wrong nonce, run, role or deployment, stale or future-dated envelope | `INSUFFICIENT_EVIDENCE` |
| A `--required-fact` the collector could not read | `INSUFFICIENT_EVIDENCE` |
| Comparable facts differ between the two snapshots (a restart, a rolling deploy, a different replica behind a load balancer) | `INSUFFICIENT_EVIDENCE` |
| A collected fact contradicts the declaration (for example declared `kv_cache_dtype=fp8`, observed `auto`) | `INSUFFICIENT_EVIDENCE` |
| A collected fact differs between reference and candidate without being declared | `NOT_COMPARABLE` |

Per-host identifiers (GPU UUID hashes, the model path hash and
`CUDA_VISIBLE_DEVICES`) are kept as receipts and excluded from comparison, so
servers on different hosts or GPUs can still be compared.

Both runs must configure a collector, or neither. Without collectors the
profile keeps its claim-only behavior and says so in every report.

## Threat model and limits

- A valid signature proves the envelope came from a holder of a trusted key and
  was not modified in transit. It does **not** prove the facts are true: a
  compromised host or collector can sign false facts. This is
  `collector_observed`, not hardware attestation; `hardware_attested` is
  reserved for future attested-boot or confidential-computing evidence.
- HMAC keys are shared secrets: anyone who can verify can also sign. Protect
  the key ring like any credential and give each environment its own keys.
  Ed25519 public-key verification is available when the optional
  `cryptography` package is installed.
- The collector reads the process it is pointed at. If a load balancer routes
  requests to other replicas, run one collector per replica or verify a direct
  replica URL; the two-snapshot check catches a switch between snapshots, not
  per-request routing.
- Engine facts come from the server's command line. Configuration supplied
  only through a config file or environment variables the collector does not
  read is reported by default value or not at all.
