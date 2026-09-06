# AgentFrameRelay

**A compatibility and portability layer for AI agents, tools, and model providers.**

AgentFrameRelay lets you define agents and tools once and run them across different AI runtimes through a common interface.

Instead of rewriting the same agent/tool logic for every framework, AgentFrameRelay provides a framework-neutral layer based on:

- `AgentSpec`
- `ToolSpec`
- `ModelSpec`
- Runtime adapters
- Standardized results and errors
- Portable tool execution
- Capability discovery

The goal is **not** to replace LangChain, CrewAI, OpenAI Agents, Google ADK, MCP, or LiteLLM.

The goal is to make it easier to **move between them, compare them, and avoid unnecessary framework lock-in.**

---

## Why AgentFrameRelay?

AI agent frameworks are evolving quickly, but application code can become tightly coupled to whichever runtime was chosen first.

For example, the same business tool may need different wrappers for:

- LangChain
- CrewAI
- OpenAI Agents
- Google ADK
- MCP
- LiteLLM

AgentFrameRelay separates the **portable definition** from the **runtime implementation**.

```text
                    AgentFrameRelay
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
        PORTABILITY    RELIABILITY   EVALUATION
             │             │             │
         AgentSpec      Validation    Benchmarks
         ToolSpec       Errors        Metrics
         ModelSpec      Async         Comparison
             │          Execution
             └─────────────┬─────────────┘
                           │
                    Runtime Adapter
                           │
          ┌────────┬───────┼────────┬──────────┐
          ▼        ▼       ▼        ▼          ▼
      LangChain  CrewAI   MCP      Google-ADK   OpenAI Agents
```

---

# Current Status

AgentFrameRelay is currently focused on **Phase 1: Core Portability** and the foundation of **Phase 2: Reliable Tool Execution**.

### Phase 1 — Core Portability

- Portable `AgentSpec`
- Portable `ToolSpec`
- Provider-aware `ModelSpec`
- Runtime adapters
- Common `Agent.run()` interface
- Common `AgentResult`
- Tool portability across supported runtimes
- Runtime capability discovery
- Lazy runtime loading through adapter resolution

### Phase 2 — Reliable Tool Execution

The current foundation includes:

- Typed tool input validation
- Pydantic-based validation and coercion
- Standardized tool errors
- Sync tool execution
- Async tool execution
- `invoke()`
- `ainvoke()`
- Safe execution of async tools from synchronous code when no event loop is active
- Portable conversion of tools to supported frameworks

> Retry policies, lifecycle hooks, timeouts, middleware, authorization, and tracing are planned reliability features. They should not be considered part of the current public API unless explicitly implemented in the installed version.

---

# Installation

Install the core package:

```bash
pip install agentframerelay
```

Install the integrations you need using optional dependencies:

```bash
pip install "agentframerelay[langchain]"
pip install "agentframerelay[crewai]"
pip install "agentframerelay[openai]"
pip install "agentframerelay[litellm]"
pip install "agentframerelay[google-adk]"
pip install "agentframerelay[mcp]"
```

For development:

```bash
pip install "agentframerelay[dev]"
```

---

# Quick Start

## Create a Tool

```python
from agentframerelay import tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
```

The tool contains its own:

- Name
- Description
- Function
- Signature
- Type information
- Validation schema
- Sync/async information

You can execute it directly:

```python
result = add.invoke(10, 20)

print(result)
```

---

# Typed Tool Validation

Tool inputs are validated before execution.

```python
from agentframerelay import tool

@tool
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    return a / b
```

Invalid inputs are normalized into AgentFrameRelay tool errors instead of exposing framework-specific validation behavior.

The goal is to keep tool execution consistent regardless of the runtime using the tool.

---

# Async Tools

Async tools are supported directly.

```python
from agentframerelay import tool

@tool
async def fetch_data(url: str) -> str:
    """Fetch data from a URL."""
    ...
```

Use:

```python
result = await fetch_data.ainvoke("https://example.com")
```

The synchronous `invoke()` path also handles async tools safely when it is running outside an existing event loop.

---

# Standardized Tool Errors

AgentFrameRelay exposes framework-neutral tool errors.

```python
from agentframerelay.errors import (
    AgentFrameRelayError,
    ToolError,
    ToolInputError,
    ToolExecutionError,
    AsyncToolError,
)
```

This means application code can handle failures without depending directly on the exception types of a specific agent framework.

---

# Create an Agent

```python
from agentframerelay import Agent

agent = Agent(
    name="math-agent",
    runtime=RUNTIME,
    model={
        "provider": MODEL_PROVIDER,
        "model": MODEL_NAME,
        "api_key": MODEL_API_KEY,
    },
    instructions="""
    You are a helpful mathematical assistant.""",
    tools=[
        add,
        multiply,
    ],
)
```

Run it through the common API:

```python
result = agent.run("What is 10 + 20?")

print(result.output)
```

The same high-level API can be used with supported runtime adapters.

---

# Runtime Adapters

AgentFrameRelay currently provides adapters for:

| Runtime | Adapter |
|---|---|
| Mock | `mock` |
| LangChain | `langchain` |
| CrewAI | `crewai` |
| OpenAI Agents | `openai` |
| Google ADK | `google_adk` |
| LiteLLM | `litellm` |
| MCP | `mcp` |

Example:

```python
agent = Agent(
    name="research-agent",
    runtime="langchain",
    model={
        "provider": MODEL_PROVIDER,
        "model": MODEL_NAME,
        "api_key": MODEL_API_KEY,
    },
    instructions="Research the topic and summarize the findings.",
    tools=[my_tool],
)
```

Changing the runtime does not require changing the portable tool definition.

---

# Runtime vs Model Provider

A runtime and a model provider are different concepts.

The `runtime` tells AgentFrameRelay **which agent runtime to use**.

The `model` tells AgentFrameRelay **which model provider and model to use**.

Use the model configuration in this form:

```python
Agent(
    name="math-agent",
    runtime=RUNTIME,
    model={
        "provider": MODEL_PROVIDER,
        "model": MODEL_NAME,
        "api_key": MODEL_API_KEY,
    },
    instructions="You are a helpful mathematical assistant.",
    tools=[add, multiply],
)
```

For example:

```python
RUNTIME = "langchain"

MODEL_PROVIDER = "openai"
MODEL_NAME = "gpt-4o-mini"
MODEL_API_KEY = None
```

Or:

```python
RUNTIME = "crewai"

MODEL_PROVIDER = "groq"
MODEL_NAME = "llama-3.3-70b-versatile"
MODEL_API_KEY = None
```

The portable representation is backed by `ModelSpec`:

```python
from agentframerelay import ModelSpec

model = ModelSpec(
    provider=MODEL_PROVIDER,
    model=MODEL_NAME,
    api_key=MODEL_API_KEY,
    parameters={},
)
```

Provider-specific model parameters can be supplied through `parameters`.

```python
ModelSpec(
    provider="openai",
    model="gpt-4o-mini",
    api_key=None,
    parameters={
        "temperature": 0.2,
    },
)
```

# Portable Mode vs Native Mode

AgentFrameRelay intentionally supports two ideas:

### Portable mode

Use the AgentFrameRelay abstractions when portability matters.

```python
agent.run(...)
tool.invoke(...)
```

### Native mode

When you need a framework-specific feature, access the native runtime object:

```python
native_agent = agent.native()
```

This is an important design principle:

> **Portable when you want portability. Native when you need native features.**

AgentFrameRelay should not prevent developers from using capabilities that are unique to a particular runtime.

---

# Tool Portability

A single relay tool can be converted for supported runtimes.

Conceptually:

```python
@tool
def search(query: str) -> str:
    """Search for information."""
    ...
```

The same tool can be adapted to different ecosystems:

```python
search.to_langchain()
search.to_crewai()
search.to_openai()
search.to_google_adk()
search.to_mcp()
search.to_litellm()
```

This keeps the canonical tool implementation in one place while allowing runtime-specific integrations.

---

# Runtime Capabilities

Different agent runtimes provide different features.

AgentFrameRelay exposes capability information:

```python
agent.capabilities()
```

A runtime can advertise capabilities such as:

```python
{
    "streaming": True,
    "memory": True,
    "human_in_loop": False,
    "durable_execution": False,
    "multi_agent": True,
}
```

This prevents AgentFrameRelay from pretending that every runtime supports exactly the same functionality.

Applications can inspect capabilities before relying on runtime-specific behavior.

---

# Architecture

The architecture is intentionally layered.

```text
Application
     │
     ▼
AgentFrameRelay
     │
     ├── AgentSpec
     ├── ToolSpec
     ├── ModelSpec
     ├── Tool execution
     ├── Validation
     ├── Errors
     └── Capabilities
     │
     ▼
Runtime Adapter
     │
     ├── LangChain
     ├── CrewAI
     ├── OpenAI Agents
     ├── Google ADK
     ├── LiteLLM
     └── MCP
```

The portable layer should contain the application's stable intent.

The adapter should contain runtime-specific implementation details.

---

# Design Principles

## 1. Framework-neutral core

The core package should not require every supported framework.

Optional integrations are installed separately.

## 2. One canonical tool definition

The portable tool should own its:

- Schema
- Validation
- Execution
- Error semantics
- Async behavior

Adapters should translate that definition rather than duplicate business logic.

## 3. Do not hide runtime differences

LangChain, CrewAI, OpenAI Agents, Google ADK, MCP, and other runtimes have different execution models.

AgentFrameRelay exposes capabilities rather than pretending those differences do not exist.

## 4. Native escape hatch

Portability should never mean giving up access to native framework functionality.

Use:

```python
agent.native()
```

when the native runtime is required.

## 5. Optional dependencies

Installing the core package should not require installing every supported AI framework.

Runtime-specific dependencies should remain optional.

---

# Roadmap

## Phase 1 — Core Portability

**Status: Foundation implemented**

- AgentSpec
- ToolSpec
- ModelSpec
- Agent facade
- Runtime adapters
- Tool portability
- Provider-aware model representation
- Standard result representation
- Capability discovery

## Phase 2 — Reliable Tool Execution

**Status: Foundation implemented**

- Typed validation
- Input coercion
- Sync execution
- Async execution
- `invoke()`
- `ainvoke()`
- Standardized tool errors
- Safe async/sync boundaries
- Portable execution semantics

Planned additions:

- Retry policies
- Exponential backoff
- Timeouts
- Lifecycle hooks
- Middleware
- Authorization
- Execution context
- Cancellation
- More detailed failure metadata

## Phase 3 — Observability

Planned:

- Execution context
- Tool lifecycle events
- Agent lifecycle events
- Structured logging
- Metrics
- Latency tracking
- Token usage
- Cost tracking
- Tracing integration

## Phase 4 — Evaluation

A major goal of AgentFrameRelay is to make agents comparable across runtimes.

The same agent specification could eventually be evaluated using:

- Correctness
- Tool-call success
- Output quality
- Consistency
- Latency
- Cost
- Reliability
- Custom evaluation metrics

Example:

```text
                 AgentSpec
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
      LangChain    CrewAI     ADK
          │         │         │
          ▼         ▼         ▼
       Evaluate   Evaluate  Evaluate
          │         │         │
          └─────────┼─────────┘
                    ▼
              Comparison
```

## Phase 5 — Runtime Benchmarking

Eventually, AgentFrameRelay can become a practical benchmark layer for agent runtimes.

For the same workload:

```text
AgentSpec
   │
   ├── Runtime A → quality / latency / cost / reliability
   ├── Runtime B → quality / latency / cost / reliability
   └── Runtime C → quality / latency / cost / reliability
```

This can help answer:

> Which runtime performs best for this workload?

rather than assuming one framework is always best.

## Phase 6 — Production Hardening

Planned:

- Compatibility matrix
- CI matrix across supported runtimes
- Better dependency isolation
- Security review
- Stable API guarantees
- Documentation improvements
- Runtime version compatibility tracking
- Better error diagnostics

## Phase 7 — Ecosystem

Longer term:

- Community adapters
- Plugin architecture
- More runtimes
- More model providers
- Evaluation datasets
- Benchmark suites
- Runtime selection recommendations

---

# Project Structure

```text
agentframerelay/
├── src/
│   └── agentframerelay/
│       ├── agent.py
│       ├── errors.py
│       ├── runtime.py
│       ├── specs.py
│       ├── tool.py
│       └── adapters/
│           ├── crewai.py
│           ├── google_adk.py
│           ├── langgraph.py
│           ├── litellm.py
│           ├── mcp.py
│           ├── mock.py
│           └── openai_agents.py
│
├── tests/
├── examples/
├── docs/
├── pyproject.toml
└── README.md
```

---

# Development

Clone the repository and install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Compile-check the source:

```bash
python -m compileall -q src tests
```

Linting can be run with:

```bash
ruff check .
```

Framework-specific integration tests require the corresponding optional dependencies.

---

# Compatibility Philosophy

AgentFrameRelay does not aim to make every framework identical.

Instead:

```text
Common semantics
       +
Runtime-specific capabilities
       +
Native escape hatch
       =
Practical portability
```

The abstraction should be strong enough to make common operations portable, while remaining honest about the differences between runtimes.

---

# Long-Term Vision

AI frameworks will continue to change.

A useful portability layer should allow an application to keep its core agent and tool definitions stable while runtimes evolve underneath them.

The long-term vision for AgentFrameRelay is therefore:

> **Define once. Run across runtimes. Measure the differences. Choose deliberately.**

AgentFrameRelay aims to become a compatibility, reliability, and evaluation layer for the rapidly evolving AI agent ecosystem.

---

# License

MIT License.
