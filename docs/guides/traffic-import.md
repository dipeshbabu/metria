# Import production traffic as a workload

A policy is most meaningful on the prompts a deployment actually serves.
`metria recipe import-traffic` turns a production request log into a workload
JSONL for [endpoint verification](endpoint-verification.md) or any other
profile. It samples prompts deterministically and redacts secrets and personal
data before anything is written.

```bash
metria recipe import-traffic \
  --source requests.jsonl \
  --sample 200 --seed 17 \
  --secret-file internal-terms.txt \
  --redact-pattern customer_id='CUST-[0-9]{6}' \
  --nonempty-check \
  --output traffic-workload.jsonl \
  --manifest traffic-manifest.json
```

## Input

One JSON object per line. Each row's prompt is taken from `--field` (a dotted
path such as `request.body.prompt`) or, by default, from `prompt`, `input`, or
the last `user` message in `messages`. Rows that are not JSON, have no prompt,
or exceed 32,768 characters are skipped and counted.

## What is written

- **Workload** (`--output`): `{"id": "traffic-00001", "category": "traffic",
  "prompt": "<redacted text>"}` per sampled row, in source order.
  `--nonempty-check` attaches a `nonempty` task check so the workload can be
  verified directly; add real checks for meaningful pass rates.
- **Manifest** (`--manifest`): `metria.traffic_import.v1` with the source file's
  SHA-256, row counts, the seed, the rules applied, how many replacements each
  rule made, and the SHA-256 of every redacted prompt.

Original prompt text is never written, and the manifest never records what was
replaced. The hash in the manifest is of the redacted prompt, so it cannot be
used to confirm a guess about an original prompt.

## Redaction

Built-in rules replace matches with `[REDACTED:<RULE>]`:

| Rule | Matches |
| --- | --- |
| `private_key`, `jwt`, `bearer_token` | PEM private keys, JSON Web Tokens, bearer credentials |
| `anthropic_key`, `openai_key`, `aws_access_key`, `github_token`, `slack_token` | Common provider credential formats |
| `url_credentials` | URLs with a username and password |
| `email`, `phone`, `ipv4` | Contact details and addresses |
| `card_number` | 13–19 digit numbers that pass the Luhn check |
| `long_secret` | 40+ character mixed letter-and-digit tokens |

Add your own rules with `--redact-pattern NAME=REGEX` (repeatable) and literal
terms (customer names, project code names) with `--secret-file`, one per line.

After redacting each prompt, the import runs every rule and every literal
again. If anything still matches, it stops and **writes nothing**.

## Your responsibilities

- Redaction is pattern-based and **best effort**. Names, addresses, free-text
  identifiers and secrets in unusual formats are not detected unless you add
  rules or literals for them. Review the workload before sharing it or
  committing it.
- Only import logs you are permitted to use for evaluation, and keep the
  workload under the same access controls as other test data derived from
  production.
- Verification bundles retain prompts and outputs, so they inherit the
  workload's sensitivity.
