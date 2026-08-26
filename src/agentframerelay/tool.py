from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from functools import update_wrapper
from typing import Any, TypeVar, get_type_hints

from pydantic import TypeAdapter, ValidationError

from .errors import (
    AgentFrameRelayError,
    AsyncToolError,
    ToolExecutionError,
    ToolInputError,
)
from .specs import ToolSpec

F = TypeVar("F", bound=Callable[..., Any])

class Tool:
    """Framework-neutral callable tool."""

    def __init__(self, function: Callable[..., Any], *, name=None, description=None):
        self.function = function
        self.name = name or function.__name__
        self.description = description or inspect.getdoc(function) or ""
        self._signature = inspect.signature(function)
        try:
            self._type_hints = get_type_hints(function, include_extras=True)
        except (NameError, TypeError):
            self._type_hints = {}
        self.input_schema = self._build_schema()
        update_wrapper(self, function)

    def _build_schema(self):
        properties, required = {}, []
        for name, param in self._signature.parameters.items():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            annotation = self._annotation(name, param)
            properties[name] = TypeAdapter(annotation).json_schema()
            if param.default is inspect.Parameter.empty:
                required.append(name)
        schema = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return schema

    def spec(self):
        return ToolSpec(
            name=self.name, description=self.description,
            input_schema=self.input_schema, function=self.function
        )

    @property
    def is_async(self) -> bool:
        """Whether the wrapped function is declared with ``async def``."""
        return inspect.iscoroutinefunction(self.function)

    def invoke(self, *args, **kwargs):
        """Validate and execute a tool synchronously.

        Asynchronous tools are run with ``asyncio.run`` when no event loop is
        active. In asynchronous applications, use :meth:`ainvoke` instead.
        """
        if self.is_async:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self.ainvoke(*args, **kwargs))
            raise AsyncToolError(
                f"Tool '{self.name}' is asynchronous; use 'await tool.ainvoke(...)' "
                "inside an active event loop."
            )

        result = self._execute(*args, **kwargs)
        if inspect.isawaitable(result):
            raise AsyncToolError(
                f"Tool '{self.name}' returned an awaitable; use 'await tool.ainvoke(...)'."
            )
        return result

    async def ainvoke(self, *args, **kwargs):
        """Validate and execute a tool, awaiting asynchronous functions."""
        result = self._execute(*args, **kwargs)
        try:
            return await result if inspect.isawaitable(result) else result
        except AgentFrameRelayError:
            raise
        except Exception as exc:
            raise ToolExecutionError(f"Tool '{self.name}' failed: {exc}") from exc

    def __call__(self, *args, **kwargs):
        return self.ainvoke(*args, **kwargs) if self.is_async else self.invoke(*args, **kwargs)

    def _execute(self, *args, **kwargs):
        bound = self._validated_arguments(*args, **kwargs)
        try:
            return self.function(*bound.args, **bound.kwargs)
        except AgentFrameRelayError:
            raise
        except Exception as exc:
            raise ToolExecutionError(f"Tool '{self.name}' failed: {exc}") from exc

    def _validated_arguments(self, *args, **kwargs):
        try:
            bound = self._signature.bind(*args, **kwargs)
            bound.apply_defaults()
        except TypeError as exc:
            raise ToolInputError(f"Invalid arguments for tool '{self.name}': {exc}") from exc

        for name, value in bound.arguments.items():
            parameter = self._signature.parameters[name]
            if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                continue
            try:
                bound.arguments[name] = TypeAdapter(
                    self._annotation(name, parameter)
                ).validate_python(value)
            except ValidationError as exc:
                raise ToolInputError(
                    f"Invalid value for '{name}' in tool '{self.name}': {exc}"
                ) from exc
        return bound

    def _annotation(self, name, parameter):
        return self._type_hints.get(
            name,
            parameter.annotation
            if parameter.annotation is not inspect.Parameter.empty
            else Any,
        )

    def to_openai(self):
        return {"type": "function", "function": {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
        }}

    def to_langchain(self):
        from .adapters.langgraph import LangGraphAdapter
        return LangGraphAdapter.tool(self)

    def to_crewai(self):
        from .adapters.crewai import CrewAIAdapter
        return CrewAIAdapter.tool(self)

    def to_litellm(self):
        from .adapters.litellm import LiteLLMAdapter
        return LiteLLMAdapter.tool(self)


    def to_google_adk(self):
        from .adapters.google_adk import GoogleADKAdapter
        return GoogleADKAdapter.tool(self)

    def to_adk(self):
        return self.to_google_adk()

    def to_openai_agents(self):
        # The OpenAI Agents SDK accepts typed Python functions as agent tools.
        return self.function

    def to_mcp(self):
        from .adapters.mcp import MCPAdapter
        return MCPAdapter.tool(self)

    def register_mcp(self, server):
        from .adapters.mcp import MCPAdapter
        return MCPAdapter.register(self, server)

def tool(function=None, *, name=None, description=None):
    """Decorator for creating a framework-neutral Tool."""
    def decorator(fn):
        return Tool(fn, name=name, description=description)
    return decorator(function) if function is not None else decorator
