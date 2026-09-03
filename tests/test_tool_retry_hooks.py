import asyncio
import importlib

import pytest

from agentframerelay import ToolExecutionError, ToolInputError, tool


def test_sync_tool_retries_until_success_and_hooks_are_per_attempt():
    attempts = []
    events = []

    @tool(retries=2)
    def unstable(value: int) -> int:
        attempts.append(value)
        if len(attempts) < 3:
            raise ValueError("temporary")
        return value * 2

    unstable.before(lambda context: events.append(("before", context.attempt)))
    unstable.on_error(lambda context, error: events.append(("error", context.attempt)))
    unstable.after(lambda context, result: events.append(("after", context.attempt, result)))

    assert unstable("4") == 8
    assert attempts == [4, 4, 4]
    assert events == [
        ("before", 1), ("error", 1),
        ("before", 2), ("error", 2),
        ("before", 3), ("after", 3, 8),
    ]


def test_retry_exhaustion_preserves_structured_error_and_attempt_count():
    calls = 0

    @tool(retries=2)
    def broken():
        nonlocal calls
        calls += 1
        raise ValueError("boom")

    with pytest.raises(ToolExecutionError, match="Tool 'broken' failed: boom") as caught:
        broken()
    assert calls == 3
    assert isinstance(caught.value.__cause__, ValueError)


def test_validation_errors_are_not_retried():
    calls = 0

    @tool(retries=3)
    def typed(value: int):
        nonlocal calls
        calls += 1

    with pytest.raises(ToolInputError):
        typed("not-an-int")
    assert calls == 0


def test_retry_delay_supports_constant_and_exponential_backoff(monkeypatch):
    delays = []
    tool_module = importlib.import_module("agentframerelay.tool")
    monkeypatch.setattr(tool_module.time, "sleep", delays.append)

    @tool(retries=2, retry_delay=0.5, backoff="exponential")
    def broken():
        raise RuntimeError("nope")

    with pytest.raises(ToolExecutionError):
        broken()
    assert delays == [0.5, 1.0]


def test_zero_retries_means_one_attempt_and_constant_delay_is_reused(monkeypatch):
    delays = []
    tool_module = importlib.import_module("agentframerelay.tool")
    monkeypatch.setattr(tool_module.time, "sleep", delays.append)
    calls = 0

    @tool(retries=0, retry_delay=1)
    def no_retry():
        nonlocal calls
        calls += 1
        raise ValueError("nope")

    with pytest.raises(ToolExecutionError):
        no_retry()
    assert calls == 1
    assert delays == []


@pytest.mark.asyncio
async def test_async_tool_and_hooks_retry_without_blocking(monkeypatch):
    calls = 0
    events = []
    delays = []

    async def fake_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    @tool(retries=1, retry_delay=0.25)
    async def unstable():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("temporary")
        return "ok"

    async def before(context):
        events.append(("before", context.tool_name, context.arguments, context.attempt))

    async def on_error(context, error):
        events.append(("error", context.attempt, str(error)))

    async def after(context, result):
        events.append(("after", context.attempt, result))

    unstable.before(before)
    unstable.on_error(on_error)
    unstable.after(after)

    assert await unstable.ainvoke() == "ok"
    assert delays == [0.25]
    assert events == [
        ("before", "unstable", {}, 1),
        ("error", 1, "Tool 'unstable' failed: temporary"),
        ("before", "unstable", {}, 2),
        ("after", 2, "ok"),
    ]


def test_hook_failures_are_surfaced_and_error_hook_chains_original_error():
    @tool
    def successful():
        return "result"

    def after(_context, _result):
        raise RuntimeError("after failed")

    successful.after(after)
    with pytest.raises(RuntimeError, match="after failed"):
        successful()

    @tool
    def broken():
        raise ValueError("tool failed")

    def on_error(_context, _error):
        raise RuntimeError("hook failed")

    broken.on_error(on_error)
    with pytest.raises(RuntimeError, match="hook failed") as caught:
        broken()
    assert isinstance(caught.value.__cause__, ToolExecutionError)


def test_synchronous_invocation_supports_asynchronous_hooks():
    events = []

    @tool
    def value():
        return 42

    async def before(context):
        events.append(("before", context.attempt))

    async def after(context, result):
        events.append(("after", result))

    value.before(before)
    value.after(after)
    assert value() == 42
    assert events == [("before", 1), ("after", 42)]


def test_each_invocation_receives_isolated_context():
    contexts = []

    @tool
    def echo(value: int):
        return value

    echo.before(contexts.append)
    assert echo(1) == 1
    assert echo(2) == 2
    assert contexts[0] is not contexts[1]
    assert contexts[0].arguments == {"value": 1}
    assert contexts[1].arguments == {"value": 2}


@pytest.mark.asyncio
async def test_concurrent_async_invocations_keep_attempt_context_isolated():
    contexts = []

    @tool(retries=1)
    async def unstable(value: int):
        if value == 1 and not hasattr(unstable, "failed"):
            unstable.failed = True
            raise ValueError("retry once")
        await asyncio.sleep(0)
        return value

    unstable.before(lambda context: contexts.append(context))
    assert await asyncio.gather(unstable.ainvoke(1), unstable.ainvoke(2)) == [1, 2]
    assert [(item.arguments["value"], item.attempt) for item in contexts] == [
        (1, 1), (2, 1), (1, 2)
    ]
