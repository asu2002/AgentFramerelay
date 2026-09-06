from agentframerelay.adapters.langgraph import LangChainAdapter


def test_langchain_stream_delegates_to_native_agent():
    captured = {}
    sentinel = object()

    class NativeAgent:
        def stream(self, payload, **kwargs):
            captured["payload"] = payload
            captured["kwargs"] = kwargs
            return sentinel

    result = LangChainAdapter.stream(
        NativeAgent(),
        "Stream this answer.",
        stream_mode="messages",
    )

    assert result is sentinel
    assert captured == {
        "payload": {"messages": [{"role": "user", "content": "Stream this answer."}]},
        "kwargs": {"stream_mode": "messages"},
    }


def test_langchain_async_run_and_stream_delegate_to_native_agent():
    captured = {}
    result = object()
    chunks = ["first", "second"]

    class NativeAgent:
        async def ainvoke(self, payload, **kwargs):
            captured["run"] = (payload, kwargs)
            return result

        async def astream(self, payload, **kwargs):
            captured["stream"] = (payload, kwargs)
            for chunk in chunks:
                yield chunk

    async def exercise():
        agent = NativeAgent()
        run_result = await LangChainAdapter.arun(agent, "Run this.", temperature=0)
        stream_result = [
            chunk
            async for chunk in LangChainAdapter.astream(agent, "Stream this.")
        ]
        return run_result, stream_result

    import asyncio

    run_result, stream_result = asyncio.run(exercise())
    assert run_result.output is result
    assert run_result.runtime == "langchain"
    assert stream_result == chunks
    assert captured["run"][1] == {"temperature": 0}