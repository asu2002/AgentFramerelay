import importlib
import sys
from types import SimpleNamespace

import pytest

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

    def function_tool(function, **kwargs):
        created["function_tool"] = kwargs
        return function

    agents_module = SimpleNamespace(
        Agent=OpenAIAgent,
        ModelSettings=ModelSettings,
        OpenAIResponsesModel=OpenAIResponsesModel,
        function_tool=function_tool,
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
    assert created["function_tool"] == {
        "name_override": "add",
        "description_override": "Add.",
    }
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


def test_openai_agents_uses_sdk_environment_resolution_without_an_api_key():
    model = OpenAIAgentsAdapter._resolve_model(
        ModelSpec(provider="openai", model="gpt-4.1-mini")
    )

    assert model == "gpt-4.1-mini"


@pytest.mark.parametrize(
    ("provider", "expected_model"),
    [
        ("google", "gemini/gemini-2.5-flash"),
        ("groq", "groq/gemini-2.5-flash"),
        ("moonshot", "moonshot/gemini-2.5-flash"),
    ],
)
def test_openai_agents_routes_litellm_providers_without_async_openai(
    monkeypatch, provider, expected_model
):
    created = {}

    class LitellmModel:
        def __init__(self, **kwargs):
            created.update(kwargs)

    bridge = importlib.import_module("agents.extensions.models.litellm_model")
    monkeypatch.setattr(bridge, "LitellmModel", LitellmModel)

    OpenAIAgentsAdapter._resolve_model(
        ModelSpec(
            provider=provider,
            model="gemini-2.5-flash",
            api_key="google-key",
            parameters={"base_url": "https://example.test"},
        )
    )

    assert created == {
        "model": expected_model,
        "api_key": "google-key",
        "base_url": "https://example.test",
    }


def test_openai_agents_rejects_unknown_provider():
    with pytest.raises(ValueError, match="OpenAI Agents adapter does not support provider 'unknown'"):
        OpenAIAgentsAdapter._resolve_model(
            ModelSpec(provider="unknown", model="model-test", api_key="not-exposed")
        )
