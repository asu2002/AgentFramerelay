from agentframerelay import tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


print("Tool name:")
print(add.name)

print("\nRun tool:")
print(add(a=10, b=20))

print("\nNeutral ToolSpec:")
print(add.spec())

print("\nLiteLLM/OpenAI-compatible schema:")
print(add.to_litellm())