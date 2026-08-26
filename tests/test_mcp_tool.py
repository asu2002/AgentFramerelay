from agentframerelay import tool

@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

def test_mcp_tool_conversion_preserves_callable():
    native = add.to_mcp()
    assert native is add.function
    assert native(25, 75) == 100
