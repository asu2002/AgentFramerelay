# AgentFrameRelay

**Define agents and tools independently from the framework that executes them.**

## MVP

- Framework-neutral `@tool`
- Pydantic/JSON schemas
- Framework-neutral `Agent`
- Runtime adapter interface
- LangChain and CrewAI agent runtimes
- OpenAI Agents and Google ADK agent runtimes
- LiteLLM agent runtime and OpenAI-compatible tool schemas
- MCP/FastMCP tool registration
- Native escape hatch
- Capability discovery
- Optional dependencies

## Quick start

```python
from agentframerelay import Agent, tool

@tool
def search_customer(customer_id: str) -> dict:
    """Retrieve a customer."""
    return {"customer_id": customer_id}

agent = Agent(
    name="customer-agent",
    instructions="Help with customer lookup.",
    tools=[search_customer],
    runtime="langchain",
)

print(agent.run({"input": "Find customer 123"}))
```

Install an adapter only when needed:

```bash
pip install "agentframerelay[langchain]"
pip install "agentframerelay[crewai]"
pip install "agentframerelay[openai]"
pip install "agentframerelay[litellm]"
pip install "agentframerelay[google-adk]"
pip install "agentframerelay[mcp]"
```

The core is intentionally framework-neutral. Adapters isolate framework API churn.

## Core execution API

`Agent.run()` always returns an `AgentResult` with the native framework output,
the runtime name, and optional metadata. `RuntimeResult` remains available as a
backward-compatible alias.

Tools validate annotated inputs before direct execution and raise public,
framework-neutral exceptions such as `ToolInputError` and `ToolExecutionError`.
Asynchronous tools support `await tool.ainvoke(...)`; calling `tool.invoke(...)`
from synchronous code runs them safely when no event loop is active.

```python
from agentframerelay import tool

@tool
async def fetch_customer(customer_id: int) -> dict:
    """Retrieve a customer."""
    return {"customer_id": customer_id}

customer = await fetch_customer.ainvoke("123")
```

## Tool portability

An AgentFrameRelay tool retains its original typed Python function and can be
adapted for each supported tool protocol:

```python
@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

langchain_tool = add.to_langchain()
crewai_tool = add.to_crewai()
litellm_schema = add.to_litellm()
google_adk_tool = add.to_google_adk()
openai_agents_tool = add.to_openai_agents()
native_mcp_function = add.to_mcp()
```

For FastMCP, register the original function directly through the relay tool:

```python
add.register_mcp(mcp)
```

The Google ADK runtime creates an in-memory session by default. Pass
`user_id`, `session_id`, `state`, `session_service`, or `run_config` to
`Agent.run()` when your application needs to control ADK session handling.
MCP is a tool protocol integration, not an `Agent` runtime.

## Integration examples

Credential-backed smoke scripts live in `examples/integrations/`; they are kept
separate from the automated `tests/` suite. The LiteLLM, OpenAI Agents, and
Google ADK examples load credentials from `.env` and assert that the original
AgentFrameRelay tool function executed.
