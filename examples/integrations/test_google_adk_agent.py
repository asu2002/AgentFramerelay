"""Run a real Google ADK tool-call round trip with AgentFrameRelay.

Set GOOGLE_API_KEY before running:
    python test_google_adk_agent.py
"""

import os

from dotenv import load_dotenv

from _bootstrap import ensure_local_package

ensure_local_package()

from agentframerelay import Agent, tool

load_dotenv()


def required_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY before running this script.")
    return api_key


tool_calls = []


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    print(f"[AGENTFRAMERELAY TOOL CALLED] add({a}, {b})")
    tool_calls.append((a, b))
    return a + b


def final_text(event) -> str:
    """Extract the assistant text from Google ADK's native final event."""
    content = getattr(event, "content", None)
    parts = getattr(content, "parts", []) if content else []
    return "".join(part.text for part in parts if getattr(part, "text", None))


agent = Agent(
    name="google-adk-calculator",
    instructions="Always use the add tool for arithmetic. Never calculate manually.",
    model={
        "provider": "google",
        "model": os.getenv("GOOGLE_ADK_MODEL", "gemini-3.6-flash"),
        "api_key": required_api_key(),
    },
    tools=[add],
    runtime="google_adk",
)


if __name__ == "__main__":
    result = agent.run("What is 25 + 75?")

    print("\nGOOGLE ADK TOOL TEST SUCCESSFUL")
    print("Tool calls:", tool_calls)
    print("Session:", result.metadata["session_id"])
    print("Final response:", final_text(result.output))
