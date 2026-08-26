from types import SimpleNamespace

from agentframerelay import Agent, tool


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def test_litellm_runtime_executes_relay_tools(monkeypatch):
    requests = []

    def completion(**kwargs):
        requests.append(kwargs)
        if len(requests) == 1:
            call = SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(name="add", arguments='{"a": 25, "b": 75}'),
            )
            message = SimpleNamespace(content=None, tool_calls=[call])
        else:
            message = SimpleNamespace(content="100", tool_calls=[])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr("litellm.completion", completion)
    agent = Agent(
        name="calculator",
        instructions="Use tools for arithmetic.",
        model={"provider": "openai", "model": "gpt-test", "parameters": {"temperature": 0}},
        tools=[add],
        runtime="litellm",
    )

    result = agent.run("What is 25 plus 75?")

    assert result.runtime == "litellm"
    assert result.output.choices[0].message.content == "100"
    assert result.metadata["tool_calls"] == [
        {"name": "add", "arguments": {"a": 25, "b": 75}, "result": 100}
    ]
    assert requests[0]["model"] == "openai/gpt-test"
    assert requests[0]["tools"][0]["function"]["name"] == "add"
    assert requests[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "100",
    }


def test_litellm_runtime_executes_async_relay_tools(monkeypatch):
    @tool
    async def async_add(a: int, b: int) -> int:
        """Add two integers asynchronously."""
        return a + b

    calls = 0

    def completion(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            call = SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(name="async_add", arguments='{"a": 25, "b": 75}'),
            )
            message = SimpleNamespace(content=None, tool_calls=[call])
        else:
            message = SimpleNamespace(content="100", tool_calls=[])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr("litellm.completion", completion)
    agent = Agent(
        name="calculator",
        model={"provider": "openai", "model": "gpt-test"},
        tools=[async_add],
        runtime="litellm",
    )

    result = agent.run("What is 25 plus 75?")

    assert result.metadata["tool_calls"] == [
        {"name": "async_add", "arguments": {"a": 25, "b": 75}, "result": 100}
    ]
