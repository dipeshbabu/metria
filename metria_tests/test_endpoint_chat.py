"""Local chat-template rendering with a transformers-compatible environment."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime

import pytest

from metria.runtimes.endpoint_chat import ChatTemplate

pytest.importorskip("jinja2")


def _template(tmp_path, template, **extra):
    config = {"chat_template": template, "bos_token": "<s>", **extra}
    raw = json.dumps(config).encode()
    (tmp_path / "tokenizer_config.json").write_bytes(raw)
    return ChatTemplate(
        str(tmp_path), {"tokenizer_config.json": hashlib.sha256(raw).hexdigest()}
    )


def test_chatml_renders_one_user_turn_with_generation_prompt(tmp_path):
    chat = _template(
        tmp_path,
        "{{ bos_token }}{% for m in messages %}<|im_start|>{{ m.role }}\n"
        "{{ m.content }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
    )
    assert chat.render("hi") == (
        "<s><|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    )
    assert len(chat.sha256) == 64
    assert chat.evidence()["status"] == "verified"


def test_loop_controls_are_available(tmp_path):
    chat = _template(
        tmp_path,
        "{% for m in messages %}{{ m.content }}{% break %}{% endfor %}"
        "{% for i in range(3) %}{% if i == 1 %}{% continue %}{% endif %}{{ i }}{% endfor %}",
    )
    assert chat.render("x") == "x02"


def test_strftime_now_renders_the_current_date(tmp_path):
    chat = _template(tmp_path, "{{ strftime_now('%Y') }}")
    assert chat.render("x") == str(datetime.now().year)


def test_tojson_does_not_html_escape(tmp_path):
    chat = _template(tmp_path, "{{ messages[0].content | tojson }}")
    assert chat.render("<b>é & 'q'") == json.dumps("<b>é & 'q'", ensure_ascii=False)


def test_raise_exception_and_named_templates(tmp_path):
    chat = _template(
        tmp_path,
        [
            {"name": "tool_use", "template": "tools"},
            {"name": "default", "template": "{{ raise_exception('no tools') }}"},
        ],
    )
    with pytest.raises(RuntimeError, match="chat template error: no tools"):
        chat.render("x")


def test_missing_template_and_tampered_config_fail(tmp_path):
    raw = json.dumps({"bos_token": "<s>"}).encode()
    (tmp_path / "tokenizer_config.json").write_bytes(raw)
    with pytest.raises(RuntimeError, match="no chat template"):
        ChatTemplate(
            str(tmp_path), {"tokenizer_config.json": hashlib.sha256(raw).hexdigest()}
        )
    with pytest.raises(RuntimeError, match="does not match"):
        ChatTemplate(str(tmp_path), {"tokenizer_config.json": "0" * 64})
    with pytest.raises(ValueError, match="pinned tokenizer_config.json"):
        ChatTemplate(str(tmp_path), {"tokenizer.json": "0" * 64})


def test_sandbox_blocks_unsafe_attribute_access(tmp_path):
    from jinja2.exceptions import SecurityError

    chat = _template(tmp_path, "{{ messages.__class__.__mro__ }}")
    with pytest.raises(SecurityError):
        chat.render("x")
