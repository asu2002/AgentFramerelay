from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .specs import AgentSpec


@dataclass
class AgentResult:
    """Standard result returned by every AgentFrameRelay runtime."""

    output: Any
    runtime: str
    metadata: dict[str, Any] = field(default_factory=dict)


# Kept for users of the initial public API.
RuntimeResult = AgentResult


def _next_item(iterator):
    try:
        return True, next(iterator)
    except StopIteration:
        return False, None


async def _iterate_sync(iterator):
    while True:
        has_item, item = await asyncio.to_thread(_next_item, iterator)
        if not has_item:
            return
        yield item


class RuntimeAdapter(ABC):
    name = "unknown"

    @classmethod
    @abstractmethod
    def build(cls, spec: AgentSpec) -> Any:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def run(cls, native_agent: Any, input: Any, **kwargs) -> AgentResult:
        raise NotImplementedError

    @classmethod
    def stream(cls, native_agent: Any, input: Any, **kwargs):
        raise NotImplementedError(f"{cls.name} does not implement streaming yet.")

    @classmethod
    async def arun(cls, native_agent: Any, input: Any, **kwargs) -> AgentResult:
        return await asyncio.to_thread(cls.run, native_agent, input, **kwargs)

    @classmethod
    async def astream(cls, native_agent: Any, input: Any, **kwargs):
        try:
            stream = await asyncio.to_thread(cls.stream, native_agent, input, **kwargs)
        except NotImplementedError:
            yield await cls.arun(native_agent, input, **kwargs)
            return
        async for item in _iterate_sync(stream):
            yield item

    @classmethod
    def capabilities(cls):
        return {
            "streaming": False, "memory": False,
            "human_in_loop": False, "durable_execution": False,
            "multi_agent": False,
        }
