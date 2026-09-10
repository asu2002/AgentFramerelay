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


def test_agent_memory_accumulates_conversation_history():
    from agentframerelay.memory import Memory

    agent = Agent(name="calculator", strategy="react", tools=[add], runtime="mock", memory=Memory())

    agent.run("2 + 3")
    agent.run("3 + 4")

    messages = agent.memory.messages
    assert len(messages) == 4
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "2 + 3"
    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == "user"
    assert messages[2]["content"] == "3 + 4"


def test_agent_memory_extracts_string_from_langchain_message_payload():
    from agentframerelay.memory import Memory

    class FakeMessage:
        def __init__(self, content):
            self.content = content

    agent = Agent(name="assistant", runtime="langchain", memory=Memory())
    payload = {"messages": [FakeMessage("hello"), FakeMessage("hi from assistant")]}

    assert agent._assistant_text(payload) == "hi from assistant"


def test_agent_memory_extracts_string_from_google_adk_event_payload():
    from agentframerelay.memory import Memory

    class Part:
        def __init__(self, text, thought=False):
            self.text = text
            self.thought = thought

    class Content:
        def __init__(self, *texts):
            self.parts = [Part(text) for text in texts]

    class FinalEvent:
        def __init__(self, *texts):
            self.content = Content(*texts)

    agent = Agent(name="assistant", runtime="google_adk", memory=Memory())
    payload = FinalEvent("Hi", " hello from adk")

    assert agent._assistant_text(payload) == " hello from adk"


def test_agent_memory_skips_google_adk_thought_part():
    from agentframerelay.memory import Memory

    class Part:
        def __init__(self, text, thought=False):
            self.text = text
            self.thought = thought

    class Content:
        def __init__(self, *parts):
            self.parts = list(parts)

    class FinalEvent:
        def __init__(self):
            self.content = Content(
                Part("internal reasoning", thought=True),
                Part("Visible final answer"),
            )

    agent = Agent(name="assistant", runtime="google_adk", memory=Memory())
    assert agent._assistant_text(FinalEvent()) == "Visible final answer"


def test_agent_memory_extracts_text_from_google_adk_model_dump_dict():
    from agentframerelay.memory import Memory

    agent = Agent(name="assistant", runtime="google_adk", memory=Memory())
    payload = {
        "content": {
            "parts": [
                {"text": "internal reasoning", "thought": True},
                {"text": "Visible final answer", "thought": False},
            ]
        }
    }

    assert agent._assistant_text(payload) == "Visible final answer"


def test_agent_memory_uses_text_transcript_for_openai_runtime():
    from agentframerelay.memory import Memory

    agent = Agent(name="assistant", runtime="openai", memory=Memory())
    agent.memory.add_user_message("My name is Asutosh")
    payload = {"messages": [{"role": "user", "content": "What is my name?"}]}

    transcript = agent._memory_input(payload)
    assert isinstance(transcript, str)
    assert "User: My name is Asutosh" in transcript
    assert "User: What is my name?" in transcript


def test_agent_memory_uses_text_transcript_for_litellm_runtime():
    from agentframerelay.memory import Memory

    agent = Agent(name="assistant", runtime="litellm", memory=Memory())
    agent.memory.add_user_message("My name is Asutosh")
    payload = {"messages": [{"role": "user", "content": "What is my name?"}]}

    transcript = agent._memory_input(payload)
    assert isinstance(transcript, str)
    assert "User: My name is Asutosh" in transcript
    assert "User: What is my name?" in transcript


def test_agent_memory_extracts_final_output_from_openai_run_result():
    from agentframerelay.memory import Memory

    class RunResult:
        def __init__(self, final_output):
            self.final_output = final_output

    agent = Agent(name="assistant", runtime="openai", memory=Memory())
    assert agent._assistant_text(RunResult("Visible final answer")) == "Visible final answer"
