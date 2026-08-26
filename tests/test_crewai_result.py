from agentframerelay import AgentResult
from agentframerelay.adapters.crewai import CrewAIAdapter


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
