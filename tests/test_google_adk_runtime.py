from types import SimpleNamespace

from agentframerelay.adapters.google_adk import GoogleADKAdapter
from agentframerelay.specs import ModelSpec


def test_google_adk_runtime_creates_a_session_and_returns_final_event(monkeypatch):
    created = {}
    final_event = SimpleNamespace(is_final_response=lambda: True)

    class SessionService:
        async def get_session(self, **kwargs):
            created["get"] = kwargs
            return None

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
