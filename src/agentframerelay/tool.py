from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import update_wrapper
from typing import Any, Literal, TypeVar, get_type_hints

from pydantic import TypeAdapter, ValidationError

from .errors import (
    AgentFrameRelayError,
    AsyncToolError,
    ToolExecutionError,
    ToolInputError,
)
from .specs import ToolSpec

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class ToolContext:
    """Per-attempt information supplied to tool lifecycle hooks."""

    tool_name: str
    arguments: dict[str, Any]
    attempt: int = 1

class Tool:
    """Framework-neutral callable tool."""

    def __init__(
        self,
        function: Callable[..., Any],
        *,
        name=None,
        description=None,
        retries: int = 0,
        retry_delay: float = 0.0,
        backoff: Literal["constant", "exponential"] = "constant",
    ):
        if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
            raise ValueError("retries must be a non-negative integer")
        if retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")
        if backoff not in {"constant", "exponential"}:
            raise ValueError("backoff must be 'constant' or 'exponential'")
        self.function = function
        self.name = name or function.__name__
        self.description = description or inspect.getdoc(function) or ""
        self.retries = retries
        self.retry_delay = retry_delay
        self.backoff = backoff
        self._before_hooks: list[Callable[[ToolContext], Any]] = []
        self._after_hooks: list[Callable[[ToolContext, Any], Any]] = []
        self._error_hooks: list[Callable[[ToolContext, Exception], Any]] = []
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
            input_schema=self.input_schema, function=self.function, tool=self
        )

    @classmethod
    def from_spec(cls, spec: ToolSpec) -> Tool:
        """Recover a core tool from a spec, with a legacy-spec fallback."""
        return spec.tool if isinstance(spec.tool, cls) else cls(
            spec.function, name=spec.name, description=spec.description
        )

    def before(self, hook: Callable[[ToolContext], Any]) -> Callable[[ToolContext], Any]:
        """Register a hook that runs before each execution attempt."""
        self._before_hooks.append(hook)
        return hook

    def after(self, hook: Callable[[ToolContext, Any], Any]) -> Callable[[ToolContext, Any], Any]:
        """Register a hook that runs after each successful execution attempt."""
        self._after_hooks.append(hook)
        return hook

    def on_error(
        self, hook: Callable[[ToolContext, Exception], Any]
    ) -> Callable[[ToolContext, Exception], Any]:
        """Register a hook that runs after each failed execution attempt."""
        self._error_hooks.append(hook)
        return hook

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

        bound = self._validated_arguments(*args, **kwargs)
        return self._invoke_sync(bound)

    async def ainvoke(self, *args, **kwargs):
        """Validate and execute a tool, awaiting asynchronous functions."""
        bound = self._validated_arguments(*args, **kwargs)
        return await self._invoke_async(bound)

    def __call__(self, *args, **kwargs):
        return self.ainvoke(*args, **kwargs) if self.is_async else self.invoke(*args, **kwargs)

    def _invoke_sync(self, bound):
        for attempt in range(1, self.retries + 2):
            context = self._context(bound, attempt)
            self._run_sync_hooks(self._before_hooks, context)
            try:
                result = self.function(*bound.args, **bound.kwargs)
                if inspect.isawaitable(result):
                    raise AsyncToolError(
                        f"Tool '{self.name}' returned an awaitable; use 'await tool.ainvoke(...)'."
                    )
            except Exception as exc:  # noqa: BLE001 - execution failures are normalized.
                error = self._execution_error(exc)
                self._run_error_hooks(context, error)
                if self._can_retry(error, attempt):
                    self._sleep(attempt)
                    continue
                raise error
            self._run_sync_hooks(self._after_hooks, context, result)
            return result

    async def _invoke_async(self, bound):
        for attempt in range(1, self.retries + 2):
            context = self._context(bound, attempt)
            await self._run_async_hooks(self._before_hooks, context)
            try:
                result = self.function(*bound.args, **bound.kwargs)
                result = await result if inspect.isawaitable(result) else result
            except Exception as exc:  # noqa: BLE001 - execution failures are normalized.
                error = self._execution_error(exc)
                await self._run_async_error_hooks(context, error)
                if self._can_retry(error, attempt):
                    await asyncio.sleep(self._delay_for(attempt))
                    continue
                raise error
            await self._run_async_hooks(self._after_hooks, context, result)
            return result

    def _context(self, bound, attempt: int) -> ToolContext:
        return ToolContext(self.name, dict(bound.arguments), attempt)

    def _execution_error(self, exc: Exception) -> Exception:
        if isinstance(exc, AgentFrameRelayError):
            return exc
        error = ToolExecutionError(f"Tool '{self.name}' failed: {exc}")
        error.__cause__ = exc
        return error

    def _can_retry(self, error: Exception, attempt: int) -> bool:
        return isinstance(error, ToolExecutionError) and attempt <= self.retries

    def _delay_for(self, attempt: int) -> float:
        return self.retry_delay * (2 ** (attempt - 1) if self.backoff == "exponential" else 1)

    def _sleep(self, attempt: int) -> None:
        delay = self._delay_for(attempt)
        if delay:
            time.sleep(delay)

    @staticmethod
    def _await_sync(value: Any) -> Any:
        if not inspect.isawaitable(value):
            return value
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(value)
        raise AsyncToolError("An asynchronous hook requires 'await tool.ainvoke(...)' inside an active event loop.")

    def _run_sync_hooks(self, hooks, *arguments) -> None:
        for hook in hooks:
            self._await_sync(hook(*arguments))

    async def _run_async_hooks(self, hooks, *arguments) -> None:
        for hook in hooks:
            result = hook(*arguments)
            if inspect.isawaitable(result):
                await result

    def _run_error_hooks(self, context: ToolContext, error: Exception) -> None:
        try:
            self._run_sync_hooks(self._error_hooks, context, error)
        except Exception as hook_error:
            raise hook_error from error

    async def _run_async_error_hooks(self, context: ToolContext, error: Exception) -> None:
        try:
            await self._run_async_hooks(self._error_hooks, context, error)
        except Exception as hook_error:
            raise hook_error from error

    def adapter_callable(self) -> Callable[..., Any]:
        """Return a signature-preserving callable that executes this core Tool."""
        if self.is_async:
            async def invoke_adapter(*args, **kwargs):
                return await self.ainvoke(*args, **kwargs)
        else:
            def invoke_adapter(*args, **kwargs):
                return self.invoke(*args, **kwargs)
        return update_wrapper(invoke_adapter, self.function)

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
        from .adapters.langchain import LangChainAdapter
        return LangChainAdapter.tool(self)

    def to_crewai(self):
        from .adapters.crewai import CrewAIAdapter
        return CrewAIAdapter.tool(self)

    def to_litellm(self):
        from .adapters.litellm import LiteLLMAdapter
        return LiteLLMAdapter.tool(self)


    def to_google_adk(self):
        # This direct export is the established native escape hatch. Agent
        # runtime construction still uses GoogleADKAdapter.tool(), which
        # delegates back through this Tool's execution layer.
        return self.function

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

def tool(
    function=None,
    *,
    name=None,
    description=None,
    retries: int = 0,
    retry_delay: float = 0.0,
    backoff: Literal["constant", "exponential"] = "constant",
):
    """Decorator for creating a framework-neutral Tool."""
    def decorator(fn):
        return Tool(
            fn,
            name=name,
            description=description,
            retries=retries,
            retry_delay=retry_delay,
            backoff=backoff,
        )
    return decorator(function) if function is not None else decorator
