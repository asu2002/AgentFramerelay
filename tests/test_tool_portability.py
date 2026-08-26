from agentframerelay import tool


@tool
def greet(name: str, excited: bool = False) -> str:
    """Return a greeting."""
    return f"Hello, {name}{'!' if excited else '.'}"


def test_tool_schema_captures_optional_arguments():
    schema = greet.spec().input_schema

    assert schema["properties"]["name"]["type"] == "string"
    assert schema["properties"]["excited"]["type"] == "boolean"
    assert schema["required"] == ["name"]


def test_litellm_and_openai_exports_share_the_neutral_schema():
    assert greet.to_litellm()["function"]["parameters"] == greet.input_schema
    assert greet.to_openai()["function"]["parameters"] == greet.input_schema


def test_callable_based_protocol_exports_preserve_the_original_function():
    assert greet.to_google_adk() is greet.function
    assert greet.to_openai_agents() is greet.function
    assert greet.to_mcp() is greet.function
