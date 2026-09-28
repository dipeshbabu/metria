"""Offline contract coverage for the optional component adapter, without a GPU."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from kv_fidelity.backends import vllm as module
from kv_fidelity.backends.base import BackendCapabilityError


@pytest.fixture
def runtime(monkeypatch):
    instances = []
    cache_clears = []

    class Tokenizer:
        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return [10, 20, 30]

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is False and add_generation_prompt is True
            return "|".join(
                message["role"] + ":" + message["content"] for message in messages
            )

    class LLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls = []
            self.tokenizer = Tokenizer()
            instances.append(self)

        def get_tokenizer(self):
            return self.tokenizer

        def generate(self, prompts, parameters, *, use_tqdm):
            assert use_tqdm is False
            self.calls.append((prompts, parameters))
            return [
                SimpleNamespace(
                    outputs=[SimpleNamespace(text="two words", token_ids=[10, 20, 30])]
                )
            ]

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            LLM=LLM,
            SamplingParams=lambda **kwargs: kwargs,
            __version__="fixture-version",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True, empty_cache=lambda: cache_clears.append(True)
            )
        ),
    )
    monkeypatch.setattr(module, "_VLLM_LLM_CACHE", {})
    return instances, cache_clears


def test_one_instance_cache_reuses_exact_key_and_evicts_before_new_configuration(
    runtime,
):
    instances, cleared = runtime
    first = module._get_llm("model-a", "auto", 256)
    assert module._get_llm("model-a", "auto", 256) is first
    second = module._get_llm("model-b", "fp8_e4m3", 512)
    assert second is not first
    assert len(instances) == 2 and len(module._VLLM_LLM_CACHE) == 1
    assert cleared == [True]
    assert second.kwargs["kv_cache_dtype"] == "fp8_e4m3"
    assert second.kwargs["enable_prefix_caching"] is False


@pytest.mark.parametrize("temperature,expected_seed", [(0.0, None), (0.7, 19)])
def test_completion_keeps_template_roles_sampling_and_authoritative_token_count(
    runtime, monkeypatch, temperature, expected_seed
):
    instances, _ = runtime
    monkeypatch.setenv("KV_FIDELITY_VLLM_MAX_MODEL_LEN", "64")
    backend = module.VLLMBackend()
    result = backend.run_completion(
        model="model",
        prompt="hello",
        system="system",
        kv_config_str="ctk=f16,ctv=f16",
        ctx=128,
        n_predict=8,
        seed=19,
        temperature=temperature,
    )
    assert result.n_tokens == 3  # native token IDs, not the two output words
    prompts, sampling = instances[-1].calls[-1]
    assert prompts == ["system:system|user:hello"]
    assert sampling == {
        "max_tokens": 8,
        "temperature": temperature,
        "seed": expected_seed,
    }
    assert instances[-1].kwargs["max_model_len"] == 168


def test_plain_trajectory_and_tokenization_preserve_native_ids(runtime):
    instances, _ = runtime
    backend = module.VLLMBackend()
    result = backend.run_completion_trajectory(
        model="model",
        prompt="raw prompt",
        kv_config_str="ctk=f16,ctv=f16",
        apply_chat_template=False,
    )
    assert result.token_ids == [10, 20, 30]
    assert instances[-1].calls[-1][0] == ["raw prompt"]
    assert backend.tokenize_to_ids(model="model", text="different text") == [10, 20, 30]
    assert backend.model_metadata(model="model")["vllm_version"] == "fixture-version"


def test_unavailable_optional_gpu_cleanup_does_not_skip_cache_replacement(
    runtime, monkeypatch
):
    module._get_llm("one", "auto", 64)
    monkeypatch.setitem(sys.modules, "torch", None)
    module._get_llm("two", "auto", 64)
    assert list(module._VLLM_LLM_CACHE) == [("two", "auto", 64)]


def test_missing_runtime_metadata_remains_unknown(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", None)
    assert (
        module.VLLMBackend().model_metadata(model="model")["vllm_version"] == "unknown"
    )


def test_short_corpus_cannot_produce_kld_evidence(runtime, tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("fixture")
    with pytest.raises(BackendCapabilityError, match="too short"):
        module.VLLMBackend().run_kld(
            model="model",
            corpus=corpus,
            ref_kv_str="ctk=f16,ctv=f16",
            cand_kv_str="ctk=q8_0,ctv=q8_0",
            ctx=16,
            chunks=2,
        )
