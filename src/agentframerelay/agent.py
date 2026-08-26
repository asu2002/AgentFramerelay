from __future__ import annotations
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
        "langchain": "langgraph",
        "adk": "google_adk",
        "google-adk": "google_adk",
        "openai_agents": "openai",
        "openai-agents": "openai",
    }
    name = aliases.get(name, name)

    if name == "langgraph":
        from .adapters.langgraph import LangGraphAdapter
        adapter = LangGraphAdapter
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
    ):
        if isinstance(model, dict):
            model = ModelSpec(**model)

        if isinstance(model, str):
            provider, sep, model_name = model.partition(":")
            if not sep:
                provider, model_name = "openai", model
            model = ModelSpec(provider=provider, model=model_name)

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
        )
        self._native_agent = None

    def _adapter(self):
        return _load_runtime(self.spec.runtime)

    def native(self):
        if self._native_agent is None:
            self._native_agent = self._adapter().build(self.spec)
        return self._native_agent

    def run(self, input, **kwargs) -> RuntimeResult:
        return self._adapter().run(self.native(), input, **kwargs)

    def stream(self, input, **kwargs):
        return self._adapter().stream(self.native(), input, **kwargs)

    def capabilities(self):
        return self._adapter().capabilities()
