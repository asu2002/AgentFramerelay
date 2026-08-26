"""Run a real LiteLLM tool-call round trip with AgentFrameRelay.

Set LITELLM_API_KEY (or GROQ_API_KEY for the default provider) before running:
    python test_litellm_agent.py
"""

import os

from dotenv import load_dotenv

from _bootstrap import ensure_local_package

ensure_local_package()

from agentframerelay import Agent, tool

load_dotenv()


def required_api_key() -> str:
    api_key = (
        os.getenv("LITELLM_API_KEY")
        or os.getenv("GROQ_API_KEY")
        or os.getenv("groq_api_key")
    )
    if not api_key:
        raise RuntimeError("Set LITELLM_API_KEY or GROQ_API_KEY before running this script.")
    return api_key


tool_calls = []


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    print(f"[AGENTFRAMERELAY TOOL CALLED] add({a}, {b})")
    tool_calls.append((a, b))
    return a + b


agent = Agent(
    name="litellm-calculator",
    instructions="Always use the add tool for arithmetic. Never calculate manually.",
    model={
        "provider": os.getenv("LITELLM_PROVIDER", "groq"),
        "model": os.getenv("LITELLM_MODEL", "openai/gpt-oss-20b"),
        "api_key": required_api_key(),
        "parameters": {"temperature": 0},
    },
    tools=[add],
    runtime="litellm",
)


if __name__ == "__main__":
    result = agent.run("Use the add tool to calculate 25 + 75.")

    print("\nLITELLM TOOL TEST SUCCESSFUL")
    print("Tool calls:", result.metadata["tool_calls"])
    print("Final response:", result.output.choices[0].message.content)
