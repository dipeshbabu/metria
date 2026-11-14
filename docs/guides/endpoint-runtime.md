# Endpoint adapter (OpenAI-compatible servers)

`metria.runtimes.EndpointAdapter` (runtime name `openai_endpoint`) runs a
measurement against an inference server that is **already running**, through
its OpenAI-compatible completions API. Metria does not start or stop the
server. To compare two servers with `metria verify`, use
[remote endpoint verification](endpoint-verification.md); the adapter can also
be used directly from Python with `execute_run()`.

## Run spec

```python
from metria import RunSpec, execute_run
from metria.measurements import TokenTrajectoryProtocol
from metria.runtimes import EndpointAdapter

spec = RunSpec(
    model={
        "id": "smollm2",                        # as listed by GET /v1/models
        "tokenizer_path": "/models/smollm2-135m",
        "tokenizer_files": {"tokenizer.json": "<sha256>", "tokenizer_config.json": "<sha256>"},
    },
    runtime={
        "name": "openai_endpoint",
        "flavor": "vllm",                       # or "generic"
        "base_url_env": "METRIA_REFERENCE_URL", # environment variable names only
        "api_key_env": "METRIA_REFERENCE_KEY",  # optional
        "timeout_s": 120,
        "max_retries": 2,
    },
    scenario={"max_tokens": 16, "temperature": 0.0, "seed": 42},
    measurements=(TokenTrajectoryProtocol.name,),
)
record = execute_run(
    study_name="endpoint",
    run_id="reference",
    spec=spec,
    adapter=EndpointAdapter(),
    measurement=TokenTrajectoryProtocol(),
    measurement_config={"prompts": [{"id": "a", "prompt": "The capital of France is"}]},
    environment={},
)
```

## What it does

- **Prompts as token IDs.** For the `vllm` flavor, prompts are tokenized locally
  with the hash-pinned `tokenizer.json` (the `tokenizers` package is required)
  and sent as token IDs. The server's echoed `prompt_token_ids` must match, so
  server-side tokenization cannot silently differ.
- **Captures.** `token_ids` comes from the response's `token_ids`
  (`return_token_ids`). The staged `token_top_logprobs` capture requests
  `logprobs=20` with `return_tokens_as_token_ids` and keeps exactly the 20 most
  likely tokens per step; like other staged captures it needs the
  `metria.staged_captures` override. A missing or short response fails the run.
- **`generic` flavor.** Sends text prompts and returns text only; token
  captures are refused in preflight.
- **Timing.** Each call's latency is measured on the client clock and
  **includes the network**. Every HTTP attempt is retained with its status code
  and duration.
- **Retries.** Transport failures and HTTP 429/500/502/503/504 are retried up to
  `max_retries` times with exponential backoff; other errors fail immediately.

## Server identity is claimed, not verified

Metria cannot hash a remote server's weights or binary, so everything a server
reports about itself is recorded with authority `partial` and source
`server_reported` in `observed["identity"]`, and `observed["identity_authority"]`
is `claimed`. Aggregate identity can never become `verified`. Instead the
adapter enforces consistency:

| Check | When | Outcome |
| --- | --- | --- |
| Claims snapshot at launch equals the snapshot after the last request (`/v1/models`: id, root, `max_model_len`; vLLM `/version`) | `observe` | Recorded in `observed["identity_snapshots"]`; drift is an evidence gap |
| A claim contradicts `runtime.declared` (`engine_version`, `max_model_len`, `model_root`) | `launch`, before any prompt | The run fails |
| Server `/tokenize` disagrees with the pinned local tokenizer on a prompt | each new prompt | Tokenizer identity `mismatch`; evidence gap |
| An identity endpoint is not exposed | any | That fact stays absent or `unknown`; no gap |

`runtime.declared` is the operator's statement of what is deployed. It accepts
only `engine`, `engine_version`, `model_root`, `model_revision`,
`max_model_len`, `dtype`, `kv_cache_dtype`, `tensor_parallel_size`,
`pipeline_parallel_size` and `hardware`, as strings or integers. Facts the
server cannot report (hardware, cache dtype) are retained as declared only.

A `root` that is a filesystem path is retained only as `root_sha256`, so a
server's private paths never enter evidence. `metria.runtimes.endpoint_identity.identity_gaps()`
returns the gaps a verification profile applies.

## Security and privacy

- The URL and API key are read only from the named environment variables. They
  are never written to recipes, run records, reports or error messages; failed
  requests retain only a status code.
- `https` is required, except `http` on loopback (`localhost`, `127.0.0.1`,
  `::1`). URLs with credentials, a query or a fragment are rejected. TLS
  certificates are always verified.
- Observed endpoint identity keeps only the scheme, port and a hash of the base
  path; the host is retained only for loopback addresses.

## Limits

- `infer()` is sequential. `EndpointSession.infer_concurrent()` sends a batch
  with a bounded number of requests in flight; each thread uses its own
  connection copy, and timings are retained under
  `observed["concurrent_invocations"]`, which run comparison ignores.
- `session.reset()` supports only the `measurement` scope; an endpoint cannot
  reset a server's prefix cache.
- Verified natively against `vllm serve` from the pinned `0.30.0+cpu` wheel:
  token IDs, top-20 logprobs, `/version`, `/v1/models` and `/tokenize`
  agreement. Other servers are untested.
