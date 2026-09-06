from agentframerelay import Agent, tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def test_tool():
    assert add(a=2, b=3) == 5
    assert add.spec().input_schema["properties"]["a"]["type"] == "integer"

def test_openai_schema():
    assert add.to_openai()["function"]["name"] == "add"

def test_mock_agent():
    agent = Agent(name="calculator", strategy="react", tools=[add], runtime="mock")
    result = agent.run("2 + 3")
    assert result.runtime == "mock"
    assert result.output["agent"] == "calculator"


def test_mock_agent_supports_async_run_and_stream_fallback():
    agent = Agent(name="calculator", strategy="react", tools=[add], runtime="mock")

    async def exercise():
        result = await agent.arun("2 + 3")
        streamed = [item async for item in agent.astream("2 + 3")]
        return result, streamed

    import asyncio

    result, streamed = asyncio.run(exercise())
    assert result.runtime == "mock"
    assert len(streamed) == 1
    assert streamed[0].runtime == "mock"


def test_litellm_schema():
    exported = add.to_litellm()
    assert exported["type"] == "function"
    assert exported["function"]["name"] == "add"
