"""Run the same AgentFrameRelay agent with a selectable runtime and model.

Set AGENT_RUNTIME, AGENT_MODEL_PROVIDER, AGENT_MODEL, and AGENT_API_KEY before
running this file, or edit the defaults in the configuration block below.
"""

import os

from _bootstrap import ensure_local_package

ensure_local_package()

from agentframerelay import Agent, tool


# -----------------------------------------------------------------------------
# Configuration: change these values to switch the framework runtime or model.
# -----------------------------------------------------------------------------

RUNTIME = os.getenv("AGENT_RUNTIME", "crewai")
MODEL_PROVIDER = os.getenv("AGENT_MODEL_PROVIDER", "google")
MODEL_NAME = os.getenv("AGENT_MODEL", "gemini-3.7-flash")
MODEL_API_KEY = os.getenv("AGENT_API_KEY") or os.getenv("GOOGLE_ADK_API_KEY")

# Example runtime choices:
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


agent = Agent(
    name="math-agent",
    runtime=RUNTIME,
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
    ]
)


# The selected runtime will decide when to call add and multiply.
result = agent.run(
    "First add 25 and 75, then multiply the result by 10."
)


print("\n" + "=" * 60)
print("AGENTFRAMERELAY RESULT")
print("=" * 60)

print(result)
