from __future__ import annotations

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
    def capabilities(cls):
        return {
            "streaming": False, "memory": False,
            "human_in_loop": False, "durable_execution": False,
            "multi_agent": False,
        }
