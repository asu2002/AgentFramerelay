from importlib import import_module


_ADAPTER_MODULES = {
    "LangChainAdapter": ".langchain",
    "LangGraphAdapter": ".langgraph",
    "CrewAIAdapter": ".crewai",
    "OpenAIAgentsAdapter": ".openai_agents",
    "GoogleADKAdapter": ".google_adk",
    "MCPAdapter": ".mcp",
}

__all__ = [
    "LangChainAdapter",
    "LangGraphAdapter",
    "CrewAIAdapter",
    "OpenAIAgentsAdapter",
    "GoogleADKAdapter",
    "MCPAdapter",
]


def __getattr__(name):
    module_name = _ADAPTER_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    adapter = getattr(import_module(module_name, __name__), name)
    globals()[name] = adapter
    return adapter
