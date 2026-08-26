"""Run a real OpenAI Agents SDK tool-call round trip with AgentFrameRelay.

Set OPENAI_API_KEY before running:
    python test_openai_agents_agent.py
"""

import os
from dotenv import load_dotenv
from _bootstrap import ensure_local_package
ensure_local_package()
from agentframerelay import Agent, tool

load_dotenv()


def required_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY before running this script.")
    return api_key


tool_calls = []


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    print(f"[AGENTFRAMERELAY TOOL CALLED] add({a}, {b})")
    tool_calls.append((a, b))
    return a + b


agent = Agent(
    name="openai-calculator",
    instructions="Always use the add tool for arithmetic. Never calculate manually.",
    model={
        "provider": "openai",
        "model": os.getenv("OPENAI_AGENTS_MODEL", "gpt-4.1-mini"),
        "api_key": required_api_key(),
        "parameters": {"temperature": 0},
    },
    tools=[add],
    runtime="openai",
)


if __name__ == "__main__":
    result = agent.run("Use the add tool to calculate 25 + 75.")
    print("\nOPENAI AGENTS TOOL TEST SUCCESSFUL")
    print("Final response:", result.output.final_output)
