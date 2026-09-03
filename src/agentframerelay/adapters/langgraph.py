from ..runtime import RuntimeAdapter, RuntimeResult
from ..specs import AgentSpec
from ..tool import Tool

class LangChainAdapter(RuntimeAdapter):
    name = "langchain"

    @classmethod
    def tool(cls, tool: Tool):
        try:
            from langchain_core.tools import StructuredTool
        except ImportError as exc:
            raise ImportError(
                "Install with: pip install 'agentframerelay[langchain]'"
            ) from exc
        return StructuredTool.from_function(
            func=tool.adapter_callable(), name=tool.name, description=tool.description
        )

    @classmethod
    def build(cls, spec: AgentSpec):
        try:
            from langchain.agents import create_agent
        except ImportError as exc:
            raise ImportError(
                "Install with: pip install 'agentframerelay[langchain]'"
            ) from exc
        if not spec.model:
            raise ValueError("A model is required for the LangChain adapter.")
        model = _resolve_model(spec.model)
        tools = [cls.tool(Tool.from_spec(t)) for t in spec.tools]
        return create_agent(model=model, tools=tools, system_prompt=spec.instructions)

    @classmethod
    def run(cls, native_agent, input, **kwargs):
        payload = input if isinstance(input, dict) else {
            "messages": [{"role": "user", "content": str(input)}]
        }
        return RuntimeResult(
            output=native_agent.invoke(payload, **kwargs), runtime=cls.name
        )

    @classmethod
    def capabilities(cls):
        return {"streaming": True, "memory": True, "human_in_loop": True,
                "durable_execution": True, "multi_agent": True}


LangGraphAdapter = LangChainAdapter

def _resolve_model(model_spec):

    try:
        from langchain_litellm import ChatLiteLLM
    except ImportError as exc:
        raise ImportError(
            "Install LiteLLM support."
        ) from exc

    kwargs = dict(model_spec.parameters or {})
    if getattr(model_spec, "api_key", None):
        kwargs["api_key"] = model_spec.api_key

    provider = (model_spec.provider or "").strip().lower()
    if provider in {"google", "google_ai", "gemini"}:
        provider = "gemini"

    return ChatLiteLLM(
        model=f"{provider}/{model_spec.model}",
        **kwargs
    )
