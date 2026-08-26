import pytest

from agentframerelay import (
    Agent,
    AgentResult,
    AsyncToolError,
    RuntimeResult,
    ToolExecutionError,
    ToolInputError,
    tool,
)


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
async def double(value: int) -> int:
    """Double an integer asynchronously."""
    return value * 2


def test_tool_validates_and_coerces_inputs():
    assert add("25", 75) == 100

    with pytest.raises(ToolInputError, match="Invalid value for 'a'"):
        add(a="not-a-number", b=75)

    with pytest.raises(ToolInputError, match="missing a required argument"):
        add(a=25)


def test_tool_wraps_execution_errors():
    @tool
    def broken() -> None:
        raise ValueError("boom")

    with pytest.raises(ToolExecutionError, match="Tool 'broken' failed: boom"):
        broken()


@pytest.mark.asyncio
async def test_async_tools_support_awaitable_invocation():
    assert double.is_async
    assert await double(50) == 100
    assert await double.ainvoke("50") == 100

    with pytest.raises(AsyncToolError, match="use 'await tool.ainvoke"):
        double.invoke(50)


def test_sync_invocation_runs_async_tools_without_an_event_loop():
    assert double.invoke(50) == 100


def test_agent_result_is_the_standard_runtime_result():
    result = Agent(name="calculator", runtime="mock").run("hello")

    assert isinstance(result, AgentResult)
    assert isinstance(result, RuntimeResult)
