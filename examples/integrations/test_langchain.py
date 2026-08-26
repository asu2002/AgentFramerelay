from _bootstrap import ensure_local_package

ensure_local_package()

from agentframerelay import tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

langchain_tool = add.to_langchain()

result = langchain_tool.invoke({
    "a": 10,
    "b": 20
})

print(result)  # 30
