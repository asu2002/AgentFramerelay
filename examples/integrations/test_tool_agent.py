from _bootstrap import ensure_local_package

ensure_local_package()

from agentframerelay import tool
import os
from langchain.agents import create_agent
from langchain_litellm import ChatLiteLLM


# ============================================================
# 1. CREATE TOOLS USING AGENTFRAMERELAY
# ============================================================

@tool
def add(a: int, b: int) -> int:
    """
    Add two integers together.

    Args:
        a: First integer.
        b: Second integer.
    """
    print(f"[TOOL CALLED] add({a}, {b})")

    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """
    Multiply two integers together.

    Args:
        a: First integer.
        b: Second integer.
    """
    print(f"[TOOL CALLED] multiply({a}, {b})")

    return a * b


# ============================================================
# 2. CONVERT AGENTFRAMERELAY TOOLS -> LANGCHAIN TOOLS
# ============================================================

langchain_tools = [
    add.to_langchain(),
    multiply.to_langchain(),
]


# ============================================================
# 3. CREATE MODEL USING CHATLITELLM
# ============================================================

model = ChatLiteLLM(
    model="groq/openai/gpt-oss-20b",

    # Put your actual Groq API key here temporarily
    api_key= os.environ.get("GROQ_API_KEY"),

    temperature=0,
)


# ============================================================
# 4. CREATE A NATIVE LANGCHAIN AGENT
# ============================================================

agent = create_agent(
    model=model,
    tools=langchain_tools,
    system_prompt="""
You are a helpful mathematical assistant.

IMPORTANT RULES:

1. Always use the available tools for mathematical calculations.
2. Never calculate numbers manually.
3. If the user asks for addition, use the add tool.
4. If the user asks for multiplication, use the multiply tool.
5. You can call multiple tools if necessary.
""",
)


# ============================================================
# 5. RUN THE AGENT
# ============================================================

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is 25 plus 75?"
            }
        ]
    }
)


# ============================================================
# 6. PRINT RESULT
# ============================================================

print("\n")
print("=" * 60)
print("FULL AGENT RESULT")
print("=" * 60)

print(result)

print("\n")
print("=" * 60)
print("FINAL MESSAGE")
print("=" * 60)

print(result["messages"][-1].content)
