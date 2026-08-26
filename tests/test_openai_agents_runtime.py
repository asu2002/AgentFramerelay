import sys
from types import SimpleNamespace

from agentframerelay.adapters.openai_agents import OpenAIAgentsAdapter
from agentframerelay.specs import AgentSpec, ModelSpec, ToolSpec


def test_openai_agents_build_maps_model_settings_and_api_key(monkeypatch):
    created = {}

    class ModelSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class OpenAIResponsesModel:
        def __init__(self, **kwargs):
            created["model"] = kwargs

    class OpenAIAgent:
        def __init__(self, **kwargs):
            created["agent"] = kwargs

    class AsyncOpenAI:
        def __init__(self, **kwargs):
            created["client"] = kwargs

    agents_module = SimpleNamespace(
        Agent=OpenAIAgent,
        ModelSettings=ModelSettings,
        OpenAIResponsesModel=OpenAIResponsesModel,
    )
    monkeypatch.setitem(sys.modules, "agents", agents_module)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=AsyncOpenAI))

    spec = AgentSpec(
        name="calculator",
        instructions="Use tools.",
        model=ModelSpec(
            provider="openai",
            model="gpt-test",
            api_key="test-key",
            parameters={"temperature": 0},
        ),
        tools=[
            ToolSpec(
                name="add",
                description="Add.",
                input_schema={},
                function=lambda a, b: a + b,
            )
        ],
    )

    OpenAIAgentsAdapter.build(spec)

    assert created["client"] == {"api_key": "test-key"}
    assert created["model"]["model"] == "gpt-test"
    assert created["agent"]["model_settings"].kwargs == {"temperature": 0}
    assert created["agent"]["tools"][0](25, 75) == 100


def test_openai_agents_stream_delegates_to_the_sdk(monkeypatch):
    sentinel = object()

    class Runner:
        @staticmethod
        def run_streamed(*args, **kwargs):
            assert args == ("native-agent", "hello")
            assert kwargs == {"max_turns": 2}
            return sentinel

    monkeypatch.setitem(sys.modules, "agents", SimpleNamespace(Runner=Runner))

    assert OpenAIAgentsAdapter.stream("native-agent", "hello", max_turns=2) is sentinel
