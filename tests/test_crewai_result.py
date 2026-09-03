from agentframerelay import AgentResult
from agentframerelay.adapters.crewai import CrewAIAdapter
from agentframerelay.specs import ModelSpec


def test_crewai_run_wraps_the_existing_native_kickoff_result(monkeypatch):
    captured = {}
    native_output = object()

    class Task:
        def __init__(self, **kwargs):
            captured["task"] = kwargs

    class Crew:
        def __init__(self, **kwargs):
            captured["crew"] = kwargs

        def kickoff(self):
            return native_output

    monkeypatch.setattr("agentframerelay.adapters.crewai.Task", Task)
    monkeypatch.setattr("agentframerelay.adapters.crewai.Crew", Crew)

    result = CrewAIAdapter.run(
        "native-agent",
        "Calculate 25 + 75.",
        expected_output="100",
        verbose=False,
    )

    assert isinstance(result, AgentResult)
    assert result.runtime == "crewai"
    assert result.output is native_output
    assert captured["task"]["agent"] == "native-agent"
    assert captured["task"]["expected_output"] == "100"
    assert captured["crew"]["verbose"] is False


def test_crewai_resolves_google_models_for_litellm(monkeypatch):
    created = {}

    class LLM:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr("crewai.LLM", LLM)

    CrewAIAdapter._resolve_model(
        ModelSpec(
            provider="google",
            model="gemini-3.6-flash",
            api_key="google-key",
            parameters={"temperature": 0},
        )
    )

    assert created == {
        "model": "gemini/gemini-3.6-flash",
        "api_key": "google-key",
        "temperature": 0,
    }
