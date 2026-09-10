from __future__ import annotations
from .memory import Memory
from .runtime import RuntimeAdapter, RuntimeResult
from .specs import AgentSpec, ModelSpec
from .tool import Tool

_RUNTIME_ADAPTERS = {}

def register_runtime(name: str, adapter: type[RuntimeAdapter]):
    _RUNTIME_ADAPTERS[name] = adapter

def _load_runtime(name):
    if name in _RUNTIME_ADAPTERS:
        return _RUNTIME_ADAPTERS[name]

    aliases = {
        "langgraph": "langchain",
        "adk": "google_adk",
        "google-adk": "google_adk",
        "openai_agents": "openai",
        "openai-agents": "openai",
    }
    name = aliases.get(name, name)

    if name == "langchain":
        from .adapters.langchain import LangChainAdapter
        adapter = LangChainAdapter
    elif name == "crewai":
        from .adapters.crewai import CrewAIAdapter
        adapter = CrewAIAdapter
    elif name == "openai":
        from .adapters.openai_agents import OpenAIAgentsAdapter
        adapter = OpenAIAgentsAdapter
    elif name == "google_adk":
        from .adapters.google_adk import GoogleADKAdapter
        adapter = GoogleADKAdapter
    elif name == "litellm":
        from .adapters.litellm import LiteLLMAdapter
        adapter = LiteLLMAdapter
    elif name == "mock":
        from .adapters.mock import MockAdapter
        adapter = MockAdapter
    else:
        raise ValueError(f"Unknown runtime: {name}")

    register_runtime(name, adapter)
    return adapter


class Agent:
    """Framework-neutral agent facade."""

    def __init__(
        self,
        *,
        name,
        instructions="",
        strategy="default",
        model=None,
        tools=None,
        runtime="mock",
        metadata=None,
        role: str | None = None,
        goal: str | None = None,
        memory: Memory | None = None,
    ):
        if isinstance(model, dict):
            model = ModelSpec(**model)

        if isinstance(model, str):
            provider, sep, model_name = model.partition(":")
            if not sep:
                provider, model_name = "openai", model
            model = ModelSpec(provider=provider, model=model_name)

        self.memory = memory or Memory()
        self.spec = AgentSpec(
            name=name,
            instructions=instructions,
            strategy=strategy,
            model=model,
            role=role,
            goal=goal,
            tools=[t.spec() if isinstance(t, Tool) else t for t in (tools or [])],
            runtime=runtime,
            metadata=metadata or {},
            memory=self.memory,
        )
        self._native_agent = None

    def _adapter(self):
        return _load_runtime(self.spec.runtime)

    def native(self):
        if self._native_agent is None:
            self._native_agent = self._adapter().build(self.spec)
        return self._native_agent

    def _memory_input(self, input):
        if not self.memory.messages:
            return input

        if isinstance(input, dict) and "messages" in input:
            messages = list(input["messages"])
        elif isinstance(input, list):
            messages = list(input)
        else:
            text = input.get("input", input) if isinstance(input, dict) else input
            messages = [{"role": "user", "content": str(text)}]

        if self.spec.runtime in {"crewai", "mock", "openai", "litellm"}:
            history = list(self.memory.messages) + messages
            content = "\n".join(
                f"{msg.get('role', 'user').title()}: {msg.get('content', '')}"
                for msg in history
            )
            return content

        return {"messages": list(self.memory.messages) + messages}

    @staticmethod
    def _assistant_text(value):
        if value is None:
            return ""
        if isinstance(value, str):
            return value

        if hasattr(value, "model_dump"):
            try:
                value = value.model_dump(exclude_none=True)
            except TypeError:
                pass

        if isinstance(value, dict):
            if isinstance(value.get("parts"), list):
                for part in reversed(value["parts"]):
                    if isinstance(part, dict) and part.get("thought") is True:
                        continue
                    extracted = Agent._assistant_text(part)
                    if extracted:
                        return extracted

            for key in ("output", "message", "content", "text", "response"):
                if key in value:
                    extracted = Agent._assistant_text(value[key])
                    if extracted:
                        return extracted
            if isinstance(value.get("messages"), list):
                for item in reversed(value["messages"]):
                    extracted = Agent._assistant_text(item)
                    if extracted:
                        return extracted
            return ""

        if isinstance(value, (list, tuple, set)):
            for item in reversed(list(value)):
                extracted = Agent._assistant_text(item)
                if extracted:
                    return extracted
            return ""

        for attr in ("content", "output", "message", "text", "response", "final_response", "final_output"):
            if hasattr(value, attr):
                extracted = Agent._assistant_text(getattr(value, attr))
                if extracted:
                    return extracted

        if hasattr(value, "parts"):
            for part in reversed(list(getattr(value, "parts", []) or [])):
                if getattr(part, "thought", False):
                    continue
                extracted = Agent._assistant_text(part)
                if extracted:
                    return extracted

        return ""

    def _record_memory(self, input, result):
        if self.memory is None:
            return

        if isinstance(input, dict) and "messages" in input:
            last_user_message = input["messages"][-1]
            if isinstance(last_user_message, dict) and last_user_message.get("role") == "user":
                self.memory.add_user_message(last_user_message.get("content", ""))
        elif not isinstance(input, list):
            text = input.get("input", input) if isinstance(input, dict) else input
            self.memory.add_user_message(text)

        if result is None:
            return

        if hasattr(result, "output"):
            payload = result.output
        else:
            payload = result

        self.memory.add_assistant_message(self._assistant_text(payload))

    def run(self, input, **kwargs) -> RuntimeResult:
        memory_input = self._memory_input(input)
        result = self._adapter().run(self.native(), memory_input, **kwargs)
        self._record_memory(input, result)
        return result

    def stream(self, input, **kwargs):
        memory_input = self._memory_input(input)
        collected = []
        for item in self._adapter().stream(self.native(), memory_input, **kwargs):
            collected.append(item)
            yield item
        if collected:
            self._record_memory(input, collected[-1])

    async def arun(self, input, **kwargs) -> RuntimeResult:
        memory_input = self._memory_input(input)
        result = await self._adapter().arun(self.native(), memory_input, **kwargs)
        self._record_memory(input, result)
        return result

    async def astream(self, input, **kwargs):
        collected = []
        async for item in self._adapter().astream(self.native(), self._memory_input(input), **kwargs):
            collected.append(item)
            yield item
        if collected:
            self._record_memory(input, collected[-1])

    def capabilities(self):
        base = self._adapter().capabilities()
        base["memory"] = base.get("memory", False) or self.memory is not None
        return base
