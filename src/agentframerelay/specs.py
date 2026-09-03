from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    description: str
    input_schema: dict[str, Any]
    function: Callable[..., Any]
    # Retains the core execution object when a spec originates from ``Tool``.
    # ``function`` remains available for manually-created and serialized specs.
    tool: Any = None

class ModelSpec(BaseModel):
    provider: str
    model: str
    api_key: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)

class AgentSpec(BaseModel):
    name: str
    instructions: str = ""
    strategy: str = "default"
    model: ModelSpec | str | None = None
    role: str | None = None
    goal: str | None = None
    tools: list[ToolSpec] = Field(default_factory=list)
    runtime: str = "mock"
    metadata: dict[str, Any] = Field(default_factory=dict)
