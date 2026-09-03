from typing import ClassVar

from ..runtime import RuntimeAdapter, RuntimeResult
from ..tool import Tool


class OpenAIAgentsAdapter(RuntimeAdapter):
    name = "openai"
    _GOOGLE_PROVIDERS: ClassVar[frozenset[str]] = frozenset({"google", "google_ai", "gemini"})

    @classmethod
    def build(cls, spec):
        try:
            from agents import Agent as OpenAIAgent
            from agents import ModelSettings, function_tool
        except ImportError as exc:
            raise ImportError(
                "Install with: pip install 'agentframerelay[openai]'"
            ) from exc
        if not spec.model:
            raise ValueError("A model is required for the OpenAI adapter.")
        agent_kwargs = {
            "name": spec.name,
            "instructions": spec.instructions,
            "model": cls._resolve_model(spec.model),
            "tools": [
                function_tool(
                    Tool.from_spec(tool_spec).adapter_callable(),
                    name_override=tool_spec.name,
                    description_override=tool_spec.description,
                )
                for tool_spec in spec.tools
            ],
        }
        model_settings = dict(spec.model.parameters)
        # ``base_url`` belongs to the SDK's LiteLLM model constructor rather
        # than ModelSettings. Other generation settings remain ModelSettings.
        if cls._provider(spec.model) != "openai":
            model_settings.pop("base_url", None)
        if model_settings:
            agent_kwargs["model_settings"] = ModelSettings(**model_settings)
        return OpenAIAgent(
            **agent_kwargs,
        )

    @classmethod
    def run(cls, native_agent, input, **kwargs):
        try:
            from agents import Runner
        except ImportError as exc:
            raise ImportError(
                "Install with: pip install 'agentframerelay[openai]'"
            ) from exc
        return RuntimeResult(
            output=Runner.run_sync(native_agent, input, **kwargs),
            runtime=cls.name,
        )

    @classmethod
    def stream(cls, native_agent, input, **kwargs):
        try:
            from agents import Runner
        except ImportError as exc:
            raise ImportError(
                "Install with: pip install 'agentframerelay[openai]'"
            ) from exc
        return Runner.run_streamed(native_agent, input, **kwargs)

    @staticmethod
    def _provider(model_spec) -> str:
        return (getattr(model_spec, "provider", "") or "").strip().lower()

    @classmethod
    def _resolve_model(cls, model_spec):
        """Resolve only model/provider combinations supported by this SDK."""
        provider = cls._provider(model_spec)
        model = model_spec.model

        if provider == "openai":
            # A bare name preserves the SDK's normal OPENAI_API_KEY behavior.
            if not model_spec.api_key:
                return model
            try:
                from agents import OpenAIResponsesModel
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise ImportError(
                    "OpenAI Agents OpenAI configuration requires the openai package. "
                    "Install with: pip install 'agentframerelay[openai]'"
                ) from exc
            return OpenAIResponsesModel(
                model=model,
                openai_client=AsyncOpenAI(api_key=model_spec.api_key),
            )

        if provider in cls._GOOGLE_PROVIDERS:
            provider = "gemini"

        try:
            import litellm
            from agents.extensions.models.litellm_model import LitellmModel
        except ImportError as exc:
            raise ImportError(
                "OpenAI Agents non-OpenAI provider support requires its LiteLLM extension. "
                "Install with: pip install 'openai-agents[litellm]'"
            ) from exc
        if provider in litellm.provider_list:
            return LitellmModel(
                model=f"{provider}/{model}",
                api_key=model_spec.api_key,
                base_url=model_spec.parameters.get("base_url"),
            )

        raise ValueError(
            "OpenAI Agents adapter does not support "
            f"provider '{model_spec.provider}' for model '{model}'."
        )

    @classmethod
    def capabilities(cls):
        return {"streaming": True, "memory": True, "human_in_loop": True,
                "durable_execution": False, "multi_agent": True}
