from __future__ import annotations

import json
from typing import Any

from ..runtime import RuntimeAdapter, RuntimeResult
from ..specs import AgentSpec
from ..tool import Tool


class LiteLLMAdapter(RuntimeAdapter):
    """LiteLLM runtime and OpenAI-compatible tool schema adapter."""

    name = "litellm"

    @staticmethod
    def tool(tool: Tool) -> dict[str, Any]:
        """Return an OpenAI-compatible function tool definition.

        LiteLLM accepts OpenAI-compatible tool definitions for supported
        providers, so the neutral AgentFrameRelay schema can be reused.
        """
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }

    @staticmethod
    def tools(tools: list[Tool]) -> list[dict[str, Any]]:
        return [LiteLLMAdapter.tool(item) for item in tools]

    @classmethod
    def build(cls, spec: AgentSpec) -> AgentSpec:
        if not spec.model:
            raise ValueError("A model is required for the LiteLLM adapter.")
        return spec

    @classmethod
    def run(cls, native_agent: AgentSpec, input: Any, **kwargs) -> RuntimeResult:
        try:
            from litellm import completion
        except ImportError as exc:
            raise ImportError(
                "Install with: pip install 'agentframerelay[litellm]'"
            ) from exc

        max_tool_rounds = kwargs.pop("max_tool_rounds", 8)
        messages = cls._messages(native_agent, input)
        relay_tools = [Tool.from_spec(item) for item in native_agent.tools]
        tools = cls.tools(relay_tools)
        tools_by_name = {item.name: item for item in relay_tools}
        request = cls._request_arguments(native_agent, messages, tools, kwargs)
        tool_calls = []

        for _ in range(max_tool_rounds + 1):
            response = completion(**request)
            message = response.choices[0].message
            calls = getattr(message, "tool_calls", None) or []
            if not calls:
                return RuntimeResult(
                    output=response,
                    runtime=cls.name,
                    metadata={"tool_calls": tool_calls},
                )

            messages.append(cls._assistant_message(message))
            for call in calls:
                name = call.function.name
                arguments = json.loads(call.function.arguments or "{}")
                tool = tools_by_name.get(name)
                if tool is None:
                    raise ValueError(f"LiteLLM requested unknown tool: {name}")

                result = tool.invoke(**arguments)
                tool_calls.append({"name": name, "arguments": arguments, "result": result})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result, default=str),
                    }
                )

            request["messages"] = messages

        raise RuntimeError(
            f"LiteLLM exceeded the configured tool-call limit ({max_tool_rounds})."
        )

    @classmethod
    def stream(cls, native_agent: AgentSpec, input: Any, **kwargs):
        try:
            from litellm import completion
        except ImportError as exc:
            raise ImportError(
                "Install with: pip install 'agentframerelay[litellm]'"
            ) from exc

        messages = cls._messages(native_agent, input)
        tools = cls.tools([Tool.from_spec(item) for item in native_agent.tools])
        return completion(
            **cls._request_arguments(native_agent, messages, tools, kwargs), stream=True
        )

    @classmethod
    def capabilities(cls):
        return {
            "streaming": True,
            "memory": False,
            "human_in_loop": False,
            "durable_execution": False,
            "multi_agent": False,
        }

    @staticmethod
    def _messages(spec: AgentSpec, input: Any) -> list[dict[str, Any]]:
        if isinstance(input, dict) and "messages" in input:
            messages = list(input["messages"])
        else:
            content = input.get("input", input) if isinstance(input, dict) else input
            messages = [{"role": "user", "content": str(content)}]

        if spec.instructions:
            messages.insert(0, {"role": "system", "content": spec.instructions})
        return messages

    @staticmethod
    def _request_arguments(
        spec: AgentSpec,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        model = f"{spec.model.provider}/{spec.model.model}"
        request = {"model": model, "messages": messages, **spec.model.parameters, **kwargs}
        if spec.model.api_key:
            request["api_key"] = spec.model.api_key
        if tools:
            request["tools"] = tools
        return request

    @staticmethod
    def _assistant_message(message: Any) -> dict[str, Any]:
        if hasattr(message, "model_dump"):
            return message.model_dump(exclude_none=True)
        if isinstance(message, dict):
            return {key: value for key, value in message.items() if value is not None}
        return {
            "role": "assistant",
            "content": getattr(message, "content", None),
            "tool_calls": getattr(message, "tool_calls", None),
        }
