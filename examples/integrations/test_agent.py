from _bootstrap import ensure_local_package

ensure_local_package()

from agentframerelay import Agent, tool
import os


# ============================================================
# 1. CREATE TOOLS USING AGENTFRAMERELAY
# ============================================================

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


# ============================================================
# 2. CREATE AGENT USING AGENTFRAMERELAY
# ============================================================

agent = Agent(
    name="math-agent",
    runtime="langgraph",
    model={
        "provider": "groq",
        "model": "openai/gpt-oss-20b",
        "api_key": os.environ.get("GROQ_API_KEY"),
        "parameters": {
            "temperature": 0
        }
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


# ============================================================
# 3. RUN AGENT
# ============================================================

result = agent.run(
    "First add 25 and 75, then multiply the result by 10."
)


# ============================================================
# 4. PRINT RESULT
# ============================================================

print("\n" + "=" * 60)
print("AGENTFRAMERELAY RESULT")
print("=" * 60)

print(result)
