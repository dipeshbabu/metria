"""Bounded-concurrency streaming through vLLM's native serving-engine API."""

from __future__ import annotations

import asyncio
import importlib
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import aclosing
from typing import Any

from .._freeze import freeze_mapping
from ..protocols import CaptureRequest, InferenceBatch, InferenceRequest, RuntimeSession
from .vllm import VLLMAdapter, VLLMSession, _native_token_ids


class AsyncEngineFacade:
    """Own one loop and expose the existing session's synchronous engine boundary."""

    def __init__(self, kwargs: Mapping[str, Any], concurrency: int) -> None:
        if type(concurrency) is not int or concurrency not in {1, 2}:
            raise ValueError("qualified serving concurrency must be 1 or 2")
        self.concurrency = concurrency
        self.last_batch: dict[str, Any] = {}
        self._loop = asyncio.new_event_loop()
        try:
            self._engine = self._loop.run_until_complete(self._create(kwargs))
        except BaseException:
            self._loop.close()
            raise
        self.model_config = self._engine.model_config
        self.llm_engine = self._engine

    async def _create(self, kwargs: Mapping[str, Any]) -> Any:
        arguments = importlib.import_module("vllm.engine.arg_utils").AsyncEngineArgs
        engine = importlib.import_module("vllm.v1.engine.async_llm").AsyncLLM
        return engine.from_engine_args(
            arguments(**dict(kwargs), disable_log_stats=False)
        )

    def get_tokenizer(self) -> Any:
        return self._engine.get_tokenizer()

    def generate(
        self,
        prompts: Sequence[str],
        *,
        sampling_params: Sequence[Any],
        use_tqdm: bool = False,
    ) -> list[Any]:
        self.last_batch = {}
        if not prompts or len(prompts) != len(sampling_params):
            raise ValueError("serving requests and sampling settings must be complete")
        return self._loop.run_until_complete(self._batch(prompts, sampling_params))

    async def _batch(self, prompts: Sequence[str], params: Sequence[Any]) -> list[Any]:
        semaphore = asyncio.Semaphore(self.concurrency)
        active = peak = 0
        traces: list[dict[str, Any] | None] = [None] * len(prompts)
        identifier = uuid.uuid4().hex

        async def request(index: int) -> Any:
            nonlocal active, peak
            async with semaphore:
                active += 1
                peak = max(peak, active)
                try:
                    response, trace = await self._stream(
                        prompts[index], params[index], f"metria-{identifier}-{index}"
                    )
                    traces[index] = trace
                    return response
                finally:
                    active -= 1

        started = time.perf_counter_ns()
        tasks = [asyncio.create_task(request(index)) for index in range(len(prompts))]
        try:
            responses = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        duration = (time.perf_counter_ns() - started) / 1_000_000_000
        self.last_batch = {
            "schema": "metria.vllm_serving_batch.v1",
            "clock": "perf_counter_ns",
            "transport": "local AsyncLLM.generate",
            "concurrency_limit": self.concurrency,
            "peak_inflight": peak,
            "window_seconds": duration,
            "requests": traces,
        }
        return list(responses)

    async def _stream(
        self, prompt: str, params: Any, request_id: str
    ) -> tuple[Any, dict[str, Any]]:
        started = time.perf_counter_ns()
        previous: tuple[int, ...] = ()
        chunks = []
        final = None
        async with aclosing(
            self._engine.generate(prompt, params, request_id)
        ) as stream:
            async for response in stream:
                elapsed = (time.perf_counter_ns() - started) / 1_000_000_000
                outputs = getattr(response, "outputs", None)
                if not outputs or len(outputs) != 1:
                    raise RuntimeError(
                        "serving stream requires one native output per request"
                    )
                tokens = _native_token_ids(
                    getattr(outputs[0], "token_ids", None), required=True
                )
                assert tokens is not None
                if (
                    tokens[: len(previous)] != previous
                    or len(tokens) > params.max_tokens
                ):
                    raise RuntimeError(
                        "serving stream changed retained tokens or exceeded its generation bound"
                    )
                added = len(tokens) - len(previous)
                if added:
                    chunks.append({"offset_seconds": elapsed, "tokens": added})
                previous = tokens
                final = response
        duration = (time.perf_counter_ns() - started) / 1_000_000_000
        if (
            final is None
            or getattr(final, "finished", None) is not True
            or not previous
        ):
            raise RuntimeError(
                "serving request did not finish with native generated tokens"
            )
        return final, {
            "request_latency_seconds": duration,
            "output_tokens": len(previous),
            "chunks": chunks,
            "finished": True,
        }

    def reset_prefix_cache(self) -> bool:
        return self._loop.run_until_complete(self._engine.reset_prefix_cache())

    def collective_rpc(self, method: str, *, timeout: float = 30) -> Any:
        return self._loop.run_until_complete(
            self._engine.collective_rpc(method, timeout=timeout)
        )

    def shutdown(self) -> None:
        try:
            self._engine.shutdown(timeout=30)
        finally:
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            self._loop.close()


class ServingSession(VLLMSession):
    def __init__(
        self,
        resolved: Mapping[str, Any],
        environment: Mapping[str, Any],
        module: Any,
        llm: Any,
        runtime_artifact: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(resolved, environment, module, llm, runtime_artifact)
        scheduler = getattr(
            getattr(llm.llm_engine, "vllm_config", None), "scheduler_config", None
        )
        self._capacity = getattr(scheduler, "max_num_seqs", None)
        if (
            type(self._capacity) is not int
            or self._capacity != resolved["runtime"]["settings"]["max_num_seqs"]
        ):
            raise ValueError(
                "native serving scheduler capacity differs from requested controls"
            )

    def observation(self) -> Mapping[str, Any]:
        return freeze_mapping(
            {
                **super().observation(),
                "serving": {
                    "engine_capacity": self._capacity,
                    "transport": "local AsyncLLM.generate",
                },
            }
        )

    def infer(
        self,
        requests: Sequence[InferenceRequest],
        capture: Sequence[CaptureRequest] = (),
    ) -> InferenceBatch:
        batch = super().infer(requests, capture)
        return InferenceBatch(
            outputs=batch.outputs,
            captures=batch.captures,
            metadata={
                **batch.metadata,
                "serving": self._llm.last_batch if requests else {},
            },
        )

    def reset(self, scope: str = "measurement") -> None:
        if scope == "serving-memory":
            if self.closed:
                raise RuntimeError("vLLM session is closed")
            self._memory_reset = self._memory_rpc("metria_reset_memory")
            return
        super().reset(scope)

    def memory_snapshot(self) -> Mapping[str, Any]:
        value = self._memory_rpc("metria_memory_snapshot")
        if (
            value.get("available") is True
            and getattr(self, "_memory_reset", {}).get("reset")
            != "torch_cuda_allocator_peak"
        ):
            raise RuntimeError(
                "native CUDA peak memory was not reset before measurement"
            )
        return freeze_mapping({**value, "reset": getattr(self, "_memory_reset", {})})

    def _memory_rpc(self, method: str) -> Mapping[str, Any]:
        if self.closed:
            raise RuntimeError("vLLM session is closed")
        rows = self._llm.collective_rpc(method, timeout=30)
        if (
            not isinstance(rows, list)
            or len(rows) != 1
            or not isinstance(rows[0], Mapping)
        ):
            raise RuntimeError("native single-worker memory observation is malformed")
        return freeze_mapping(rows[0])


class ServingAdapter(VLLMAdapter):
    worker_extension_cls = "metria.runtimes.vllm_memory.MemoryWorkerExtension"

    def resolve(self, spec: Any, environment: Mapping[str, Any]) -> Mapping[str, Any]:
        result = super().resolve(spec, environment)
        return freeze_mapping(
            {**result, "serving": {"concurrency": spec.trial_policy["concurrency"]}}
        )

    def _create_llm(
        self, module: Any, kwargs: Mapping[str, Any], resolved: Mapping[str, Any]
    ) -> Any:
        return AsyncEngineFacade(kwargs, resolved["serving"]["concurrency"])

    def _create_session(
        self,
        resolved: Mapping[str, Any],
        environment: Mapping[str, Any],
        module: Any,
        llm: Any,
        artifact: Mapping[str, Any] | None,
    ) -> RuntimeSession:
        return ServingSession(resolved, environment, module, llm, artifact)
