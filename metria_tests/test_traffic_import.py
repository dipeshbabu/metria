"""Redacted, deterministic import of production prompts as a workload."""

from __future__ import annotations

import io
import json

import pytest

from metria.cli import main
from metria.traffic_import import Redactor, extract_prompt, import_traffic

SECRETS = {
    "email": "contact jane.doe@example.com today",
    "openai_key": "key " + "sk" + "-proj-ABCDEFGHIJKLMNOPQRSTUVWX please",
    "anthropic_key": "use " + "sk" + "-ant-api03-abcdefghijklmnopqrstuv now",
    "aws_access_key": "AKIA" + "IOSFODNN7EXAMPLE was leaked",
    "github_token": "gh" + "p_" + "a" * 36 + " in logs",
    "slack_token": "xo" + "xb-1234567890-abcdefghij posted",
    "jwt": "token eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N",
    "bearer_token": "Authorization: Bearer abcdefghijklmnop0123456789",
    "url_credentials": "connect postgres://admin:hunter2@db.internal:5432/app",
    "ipv4": "server 10.20.30.40 is down",
    "card_number": "card 4111 1111 1111 1111 declined",
    "phone": "call +1 415-555-0132 tomorrow",
}


def _log(tmp_path, rows):
    path = tmp_path / "traffic.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\nnot json\n")
    return path


@pytest.mark.parametrize(("rule", "text"), sorted(SECRETS.items()))
def test_builtin_rules_redact_each_secret(rule, text):
    redactor = Redactor()
    redacted = redactor.redact(text)
    assert f"[REDACTED:{rule.upper()}]" in redacted, redacted
    assert redactor.residual(redacted) == []


def test_ordinary_text_dates_and_invalid_cards_are_kept():
    redactor = Redactor()
    text = "Meeting on 2024-01-15 at 10:30 about order 1234 5678 9012 3456."
    assert redactor.redact(text) == text
    assert redactor.counts == {}


def test_import_is_deterministic_and_drops_originals(tmp_path):
    rows = [{"prompt": f"question {i} for jane.doe@example.com"} for i in range(20)]
    rows += [{"messages": [{"role": "user", "content": "hi from 10.0.0.1"}]}, {}]
    source = _log(tmp_path, rows)
    first, manifest = import_traffic(source, sample=5, seed=7)
    second, _ = import_traffic(source, sample=5, seed=7)
    other, _ = import_traffic(source, sample=5, seed=8)
    assert first == second and first != other
    assert all("jane.doe" not in row["prompt"] for row in first)
    assert manifest["selected"] == 5 and manifest["usable_rows"] == 21
    assert manifest["skipped"] == {"invalid_json": 1, "no_prompt": 1, "too_long": 0}
    assert manifest["redactions"]["email"] >= 4
    assert len(manifest["prompt_sha256"]) == 5
    assert "jane.doe" not in json.dumps(manifest)


def test_secret_file_entries_are_replaced(tmp_path):
    source = _log(tmp_path, [{"prompt": "Project Nightingale launches in Q3"}])
    rows, manifest = import_traffic(
        source, sample=1, seed=1, secrets=["Project Nightingale"]
    )
    assert rows[0]["prompt"] == "[REDACTED:SECRET] launches in Q3"
    assert manifest["redactions"] == {"secret_file": 1}


def test_residual_matches_refuse_to_write(tmp_path, monkeypatch):
    from metria import traffic_import

    source = _log(tmp_path, [{"prompt": "mail jane@example.com"}])
    # A redactor that forgets to replace leaves a residual match.
    monkeypatch.setattr(traffic_import.Redactor, "redact", lambda self, text: text)
    with pytest.raises(ValueError, match="nothing was written"):
        import_traffic(source, sample=1, seed=1)


def test_custom_patterns_and_fields(tmp_path):
    source = _log(
        tmp_path, [{"request": {"body": {"text": "customer CUST-123456 asked"}}}]
    )
    rows, manifest = import_traffic(
        source,
        sample=1,
        seed=1,
        field="request.body.text",
        extra_rules=[("customer_id", r"CUST-\d{6}")],
        checks=[{"id": "answered", "kind": "nonempty"}],
    )
    assert rows[0]["prompt"] == "customer [REDACTED:CUSTOMER_ID] asked"
    assert rows[0]["checks"] == [{"id": "answered", "kind": "nonempty"}]
    assert "customer_id" in manifest["rules"]


def test_extract_prompt_prefers_last_user_message():
    row = {
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
    }
    assert extract_prompt(row, None) == "second"
    assert extract_prompt({"input": "x"}, None) == "x"
    assert extract_prompt({"prompt": 3}, None) is None


def test_cli_writes_workload_and_manifest(tmp_path):
    source = _log(
        tmp_path,
        [{"prompt": f"q{i} " + "sk" + "-ABCDEFGHIJKLMNOPQRSTUV"} for i in range(3)],
    )
    secret_file = tmp_path / "secrets.txt"
    secret_file.write_text("q1\n")
    output, manifest = tmp_path / "workload.jsonl", tmp_path / "manifest.json"
    out = io.StringIO()
    code = main(
        [
            "recipe", "import-traffic",
            "--source", str(source), "--output", str(output),
            "--manifest", str(manifest), "--sample", "2", "--seed", "3",
            "--secret-file", str(secret_file), "--nonempty-check",
            "--redact-pattern", "ticket=TICKET-[0-9]+",
        ],
        stdout=out,
    )  # fmt: skip
    assert code == 0, out.getvalue()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 2 and all("sk-" not in row["prompt"] for row in rows)
    assert json.loads(manifest.read_text())["seed"] == 3
    assert main(
        ["recipe", "import-traffic", "--source", str(source), "--output", str(output),
         "--manifest", str(tmp_path / "m2.json"), "--sample", "1", "--seed", "1"],
        stdout=io.StringIO(), stderr=io.StringIO(),
    ) != 0  # fmt: skip
