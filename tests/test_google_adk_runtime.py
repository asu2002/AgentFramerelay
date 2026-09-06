import importlib
from copy import deepcopy
from types import SimpleNamespace

import pytest

from agentframerelay.adapters.google_adk import GoogleADKAdapter
from agentframerelay.specs import ModelSpec


def test_google_adk_runtime_creates_a_session_and_returns_final_event(monkeypatch):
    created = {}
    final_event = SimpleNamespace(is_final_response=lambda: True)

    class SessionService:
        async def get_session(self, **kwargs):
            created["get"] = kwargs

        async def create_session(self, **kwargs):
            created["create"] = kwargs
            return SimpleNamespace(id="session-1")

    class Runner:
        def __init__(self, **kwargs):
            created["runner"] = kwargs

        async def run_async(self, **kwargs):
            created["run"] = kwargs
            yield SimpleNamespace(is_final_response=lambda: False)
            yield final_event

    monkeypatch.setattr("google.adk.runners.Runner", Runner)
    service = SessionService()

    result = GoogleADKAdapter.run(
        SimpleNamespace(name="calculator"),
        "What is 25 plus 75?",
        app_name="relay-test",
        user_id="test-user",
        session_service=service,
    )

    assert result.runtime == "google_adk"
    assert result.output is final_event
    assert result.metadata == {
        "app_name": "relay-test",
        "user_id": "test-user",
        "session_id": "session-1",
        "events": 2,
    }
    assert created["create"] == {
        "app_name": "relay-test",
        "user_id": "test-user",
        "session_id": None,
        "state": None,
    }
    assert created["run"]["new_message"].parts[0].text == "What is 25 plus 75?"


def test_google_adk_model_uses_an_explicit_api_key(monkeypatch):
    created = {}

    class Gemini:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr("google.adk.models.google_llm.Gemini", Gemini)

    GoogleADKAdapter._resolve_model(
        ModelSpec(provider="google", model="gemini-test", api_key="test-key")
    )

    assert created == {
        "model": "gemini-test",
        "client_kwargs": {"api_key": "test-key"},
    }


def test_google_adk_native_name_is_a_valid_identifier():
    assert GoogleADKAdapter._native_name("google-adk-calculator") == "google_adk_calculator"
    assert GoogleADKAdapter._native_name("123 calculator") == "agent_123_calculator"


def test_google_adk_preserves_environment_resolution_for_google_without_a_key():
    assert GoogleADKAdapter._resolve_model(
        ModelSpec(provider="google", model="gemini-2.5-flash")
    ) == "gemini-2.5-flash"


@pytest.mark.parametrize(
    ("provider", "expected_model"),
    [
        ("openai", "openai/gpt-4.1-mini"),
        ("groq", "groq/gpt-4.1-mini"),
    ],
)
def test_google_adk_routes_litellm_providers_through_installed_bridge(
    monkeypatch, provider, expected_model
):
    created = {}

    class LiteLlm:
        def __init__(self, **kwargs):
            created.update(kwargs)

    module = importlib.import_module("google.adk.models.lite_llm")
    monkeypatch.setattr(module, "LiteLlm", LiteLlm)

    GoogleADKAdapter._resolve_model(
        ModelSpec(provider=provider, model="gpt-4.1-mini", api_key="provider-key")
    )

    assert created == {
        "model": expected_model,
        "api_key": "provider-key",
    }


def test_google_adk_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Google ADK adapter does not support provider 'unknown'"):
        GoogleADKAdapter._resolve_model(
            ModelSpec(provider="unknown", model="model-test", api_key="not-exposed")
        )


@pytest.mark.asyncio
async def test_google_adk_groq_model_strips_only_reasoning_history():
    class Request:
        def __init__(self):
            self.contents = [
                SimpleNamespace(
                    parts=[
                        SimpleNamespace(text="reasoning", thought=True),
                        SimpleNamespace(text="visible content", thought=False),
                    ]
                )
            ]

        def model_copy(self, *, deep):
            assert deep
            return deepcopy(self)

    class LiteLlm:
        async def generate_content_async(self, request, stream=False):
            assert not stream
            yield request

    model = GoogleADKAdapter._groq_model_class(LiteLlm)()
    requests = [request async for request in model.generate_content_async(Request())]

    assert [part.text for part in requests[0].contents[0].parts] == ["visible content"]
