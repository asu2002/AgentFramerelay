# AgentFrameRelay Architecture

The core owns neutral specifications. Adapters own framework-specific APIs.

```text
AgentSpec
   |
   +-- RuntimeAdapter
   |      +-- LangChain
   |      +-- CrewAI
   |      +-- OpenAI Agents
   |      +-- Google ADK
   |      +-- LiteLLM
   |
   +-- ModelSpec
   +-- ToolSpec
   +-- AgentResult
```

If a framework changes, its adapter should absorb the change.

The `Agent.native()` method is the escape hatch for framework-specific features.

`Tool` conversions are intentionally separate from agent runtimes. The LiteLLM
runtime exports OpenAI-compatible schemas and executes the original tool
function during its bounded tool-call loop. MCP registers that same typed
callable with an MCP/FastMCP server. This lets the AgentFrameRelay tool remain
the single source of truth across framework-native tool APIs.

The core validates direct tool invocations from their typed Python signatures.
`Tool.invoke()` supports synchronous callers, while `Tool.ainvoke()` supports
asynchronous functions and applications. Every runtime returns `AgentResult`,
whose `output` preserves its native framework result.

Future work:
- capability resolver
- runtime="auto"
- strategy resolver
- A2A
- Microsoft Agent Framework
- PydanticAI
- LlamaIndex
- AWS Strands
- observability
