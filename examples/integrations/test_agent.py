"""Run a real AgentFrameRelay agent with a selectable runtime, model, and memory.

Set AGENT_RUNTIME, AGENT_MODEL_PROVIDER, AGENT_MODEL, and AGENT_API_KEY before
running this file, or edit the defaults below. Memory is optional and is passed
into the Agent instance so the conversation state is preserved across calls.
"""

import os

from _bootstrap import ensure_local_package

ensure_local_package()

from agentframerelay import Agent, Memory, tool


# -----------------------------------------------------------------------------
# Configuration: change these values to switch the framework runtime or model.
# -----------------------------------------------------------------------------

RUNTIME = os.getenv("AGENT_RUNTIME", "crewai")
MODEL_PROVIDER = os.getenv("AGENT_MODEL_PROVIDER", "google")
MODEL_NAME = os.getenv("AGENT_MODEL", "gemini-3.7-flash")
MODEL_API_KEY = os.getenv("AGENT_API_KEY") or os.getenv("GOOGLE_ADK_API_KEY")
USE_MEMORY = os.getenv("AGENT_USE_MEMORY", "true").lower() in {"1", "true", "yes", "on"}

# Example runtime choices:
# RUNTIME = "mock"        # no external dependency, great for local testing
# RUNTIME = "crewai"
# RUNTIME = "langchain"
# RUNTIME = "litellm"
# RUNTIME = "google_adk"
# RUNTIME = "openai"

# Example model choices:
# MODEL_PROVIDER = "google"; MODEL_NAME = "gemini-3.7-flash"
# MODEL_PROVIDER = "openai"; MODEL_NAME = "gpt-4.1-mini"
# MODEL_PROVIDER = "anthropic"; MODEL_NAME = "claude-sonnet-4-20250514"
# MODEL_PROVIDER = "groq"; MODEL_NAME = "llama-3.3-70b-versatile"


# AgentFrameRelay exposes framework-neutral tools that each runtime can call.

@tool
def add(a: int, b: int) -> int:
    """Add two integers together."""

    print(f"[TOOL CALLED] add({a}, {b})")

    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers together."""

    print(f"[TOOL CALLED] multiply({a}, {b})")

    return a * b


memory = Memory() if USE_MEMORY else None

agent = Agent(
    name="math-agent",
    runtime=RUNTIME,
    memory=memory,
    model={
        "provider": MODEL_PROVIDER,
        "model": MODEL_NAME,
        "api_key": MODEL_API_KEY,
    },
    instructions="""
You are a helpful mathematical assistant.

IMPORTANT:
- Always use the available tools for mathematical calculations.
- Never calculate numbers manually.
- Use add for addition.
- Use multiply for multiplication.
- You may call multiple tools when necessary.
""",
    tools=[
        add,
        multiply,
    ],
)


# -----------------------------------------------------------------------------
# Real multi-turn memory demo: the same Agent instance is reused.
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("FIRST RUN")
print("=" * 60)
result_one = agent.run("First add 25 and 75, then multiply the result by 10.")
print(result_one)

print("\n" + "=" * 60)
print("SECOND RUN")
print("=" * 60)
result_two = agent.run("Now add 5 and 7.")
print(result_two)

if agent.memory is not None:
    print("\n" + "=" * 60)
    print("MEMORY STATE")
    print("=" * 60)
    for idx, message in enumerate(agent.memory.messages, 1):
        print(f"{idx}. {message['role']}: {message['content']}")

print("\n" + "=" * 60)
print("AGENT CAPABILITIES")
print("=" * 60)
print(agent.capabilities())
