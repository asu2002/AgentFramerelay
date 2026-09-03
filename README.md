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

## Provider-aware model resolution

`ModelSpec` remains the shared model configuration. Each runtime adapter resolves
it only through provider paths supported by its installed SDK; a provider is never
silently redirected to another service.

```python
from agentframerelay import Agent
from agentframerelay.specs import ModelSpec

google_agent = Agent(
    name="gemini-agent",
    runtime="google_adk",
    model=ModelSpec(
        provider="google",
        model="gemini-2.5-flash",
        api_key="...",  # or use Google ADK's normal environment credentials
    ),
)

openai_agent = Agent(
    name="openai-agent",
    runtime="openai",
    model=ModelSpec(
        provider="openai",
        model="gpt-4.1-mini",
        api_key="...",  # or use OPENAI_API_KEY
    ),
)
```

Google ADK uses its native `Gemini` model for the `google`, `google_ai`, and
`gemini` provider aliases. Its installed LiteLLM integration supports other
LiteLLM providers, including `openai`, `groq`, and `moonshot` (Kimi), using a
`provider/model` route. OpenAI Agents uses its native Responses model for
`openai`; every provider recognized by installed LiteLLM—including Google/Gemini,
Groq, and Moonshot/Kimi—is routed through the OpenAI Agents LiteLLM model bridge.
Unknown providers fail clearly. No provider API key is sent to an OpenAI client
unless the provider is explicitly `openai`.

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

## Retries and lifecycle hooks

Tools can retry execution failures in the framework-neutral core. `retries` is
the number of retries after the initial call, so `retries=3` permits four total
attempts. Input validation failures and hook failures are never retried.

```python
@tool(retries=3)
def fetch_customer(customer_id: int) -> dict:
    ...

@tool(retries=3, retry_delay=1.0)
def fetch_with_delay(customer_id: int) -> dict:
    ...

@tool(retries=3, retry_delay=1.0, backoff="exponential")
def fetch_with_backoff(customer_id: int) -> dict:
    ...
```

`backoff` defaults to `"constant"`; exponential delays are 1x, 2x, 4x, and
so on. Hooks are scoped to a single `Tool` instance and receive a `ToolContext`
containing the tool name, validated arguments, and current attempt number.

```python
def before(context):
    print(f"starting {context.tool_name}, attempt {context.attempt}")

def after(context, result):
    print("completed", result)

def on_error(context, error):
    print("failed", error)

fetch_with_backoff.before(before)
fetch_with_backoff.after(after)
fetch_with_backoff.on_error(on_error)
```

Hooks run for every attempt in this order: `before`, execution, then `after`
on success or `on_error` on failure. Both normal and `async def` hooks are
supported. Use `await tool.ainvoke(...)` in async applications; asynchronous
hooks and retry waits then use the event loop without blocking it. A failing
`before` or `after` hook is surfaced directly. A failing `on_error` hook is
raised with the original tool error as its cause.

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
