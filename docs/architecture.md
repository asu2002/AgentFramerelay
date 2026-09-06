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

`ModelSpec` is resolved by the selected adapter, not by the shared core:

```text
ModelSpec(provider, model, api_key, parameters)
    |
    +-- OpenAI Agents: openai -> native Responses model
    |                    LiteLLM provider -> OpenAI Agents LiteLLM bridge
    |
    +-- Google ADK: google/gemini -> native Gemini model
                     supported LiteLLM provider -> Google ADK LiteLlm model
```

The OpenAI Agents adapter uses its native model for `openai`. Every provider
recognized by installed LiteLLM is routed through the SDK's LiteLLM bridge; the
Google aliases `google`, `google_ai`, and `gemini` normalize to `gemini/<model>`
rather than creating an OpenAI client with a Google credential. Google ADK uses
its native Gemini path for those Google aliases and its installed LiteLLM
integration for providers recognized by LiteLLM (including OpenAI, Groq, and
Moonshot/Kimi). Unsupported provider/model combinations raise an explicit error.

`Tool` conversions are intentionally separate from agent runtimes. The LiteLLM
runtime exports OpenAI-compatible schemas and executes the original tool
function during its bounded tool-call loop. MCP registers that same typed
callable with an MCP/FastMCP server. This lets the AgentFrameRelay tool remain
the single source of truth across framework-native tool APIs.

The core validates direct tool invocations from their typed Python signatures.
`Tool.invoke()` supports synchronous callers, while `Tool.ainvoke()` supports
asynchronous functions and applications. Every runtime returns `AgentResult`,
whose `output` preserves its native framework result.

Tool execution is also the single framework-neutral lifecycle boundary:

```text
validation -> before hook -> execution -> after hook / on_error hook -> retry
```

Retries operate only on wrapped execution failures, and lifecycle hooks run per
attempt. Agent-created runtime tools retain their originating `Tool` in
`ToolSpec`, so adapters delegate execution to this same core boundary rather
than implementing framework-specific retry or hook behavior. Direct native
callable exports remain available as the existing escape hatch.

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
