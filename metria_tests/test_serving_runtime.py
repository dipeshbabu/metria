import asyncio
import sys
from types import SimpleNamespace

import pytest

from metria.runtimes.vllm_memory import MemoryWorkerExtension
from metria.runtimes.vllm_serving import AsyncEngineFacade, ServingSession


class Stream:
    model_config = SimpleNamespace()

    def __init__(self, mode="normal"):
        self.mode, self.active, self.peak, self.closed = mode, 0, 0, False

    async def generate(self, prompt, params, request_id):
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            for index in range(1, 4):
                await asyncio.sleep(0)
                if self.mode == "error" and prompt == "bad":
                    raise RuntimeError("native failure")
                tokens = list(range(index))
                if self.mode == "changed" and index == 2:
                    tokens = [99, 1]
                yield SimpleNamespace(
                    finished=index == 3 and self.mode != "unfinished",
                    outputs=[]
                    if self.mode == "missing"
                    else [SimpleNamespace(token_ids=tokens, text=prompt)],
                )
        finally:
            self.active -= 1

    async def reset_prefix_cache(self):
        return True

    async def collective_rpc(self, method, timeout):
        assert timeout == 30
        return [{"method": method}]

    def get_tokenizer(self):
        return "tokenizer"

    def shutdown(self, timeout):
        assert timeout == 30
        self.closed = True


def facade(monkeypatch, engine, concurrency=2):
    async def create(self, kwargs):
        return engine

    monkeypatch.setattr(AsyncEngineFacade, "_create", create)
    return AsyncEngineFacade({}, concurrency)


@pytest.mark.parametrize("concurrency", [1, 2])
def test_stream_orders_responses_bounds_concurrency_and_closes_loop(
    monkeypatch, concurrency
):
    engine = Stream()
    client = facade(monkeypatch, engine, concurrency)
    rows = client.generate(
        ["one", "two", "three"], sampling_params=[SimpleNamespace(max_tokens=3)] * 3
    )
    assert [row.outputs[0].text for row in rows] == ["one", "two", "three"]
    assert engine.peak == client.last_batch["peak_inflight"] == concurrency
    assert all(row["output_tokens"] == 3 for row in client.last_batch["requests"])
    assert client.get_tokenizer() == "tokenizer"
    assert client.reset_prefix_cache() is True
    assert client.collective_rpc("named") == [{"method": "named"}]
    client.shutdown()
    assert client._loop.is_closed() and engine.closed


@pytest.mark.parametrize("mode", ["changed", "unfinished", "missing", "error"])
def test_invalid_stream_cancels_sibling_tasks_and_retains_no_measurement(
    monkeypatch, mode
):
    engine = Stream(mode)
    client = facade(monkeypatch, engine)
    with pytest.raises(RuntimeError):
        client.generate(
            ["bad", "two"], sampling_params=[SimpleNamespace(max_tokens=3)] * 2
        )
    assert not client.last_batch
    assert engine.active == 0
    client.shutdown()


def test_generation_limit_and_input_lengths_are_enforced(monkeypatch):
    client = facade(monkeypatch, Stream())
    with pytest.raises(ValueError):
        client.generate(["one"], sampling_params=[])
    with pytest.raises(RuntimeError, match="bound"):
        client.generate(["one"], sampling_params=[SimpleNamespace(max_tokens=1)])
    client.shutdown()
    with pytest.raises(ValueError):
        AsyncEngineFacade({}, True)


def test_memory_counts_unique_native_kv_storages_and_resets_peaks(monkeypatch):
    device = SimpleNamespace(type="cuda")
    calls = []

    class Tensor:
        def __init__(self, ptr, size):
            self.device, self.ptr, self.size = device, ptr, size

        def untyped_storage(self):
            return SimpleNamespace(data_ptr=lambda: self.ptr, nbytes=lambda: self.size)

    torch = SimpleNamespace(
        Tensor=Tensor,
        cuda=SimpleNamespace(
            synchronize=lambda d: calls.append("sync"),
            reset_peak_memory_stats=lambda d: calls.append("reset"),
            max_memory_allocated=lambda d: 100,
            max_memory_reserved=lambda d: 200,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    worker = MemoryWorkerExtension()
    worker.device = device
    worker.model_runner = SimpleNamespace(
        kv_caches=[Tensor(1, 20), Tensor(1, 20), Tensor(2, 30)]
    )
    assert worker.metria_reset_memory()["reset"] == "torch_cuda_allocator_peak"
    assert worker.metria_memory_snapshot()["allocated_kv_bytes"] == 50
    assert calls == ["sync", "reset", "sync"]
    worker.model_runner.kv_caches = []
    assert worker.metria_memory_snapshot()["available"] is False
    worker.model_runner.kv_caches = [object()]
    assert worker.metria_memory_snapshot()["available"] is False
    worker.device = SimpleNamespace(type="cpu")
    assert worker.metria_reset_memory()["available"] is False
    assert worker.metria_memory_snapshot()["available"] is False


def test_memory_session_requires_reset_confirmation_and_single_worker():
    session = object.__new__(ServingSession)
    session._closed = False
    rows = [{"available": True}]
    session._llm = SimpleNamespace(collective_rpc=lambda *a, **kw: rows)
    with pytest.raises(RuntimeError, match="reset"):
        session.memory_snapshot()
    rows[:] = [{"available": True, "reset": "torch_cuda_allocator_peak"}]
    session.reset("serving-memory")
    assert session.memory_snapshot()["reset"]["reset"] == "torch_cuda_allocator_peak"
    rows.clear()
    with pytest.raises(RuntimeError, match="malformed"):
        session.memory_snapshot()
    session._closed = True
    with pytest.raises(RuntimeError, match="closed"):
        session.reset("serving-memory")
    with pytest.raises(RuntimeError, match="closed"):
        session.memory_snapshot()
