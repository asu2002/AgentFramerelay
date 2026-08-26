from agentframerelay import tool

@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

def test_openai_agents_tool_conversion_preserves_callable():
    native = add.to_openai_agents()
    assert native is add.function
    assert native(25, 75) == 100
