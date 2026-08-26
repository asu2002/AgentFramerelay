from ..runtime import RuntimeAdapter, RuntimeResult


class OpenAIAgentsAdapter(RuntimeAdapter):
    name = "openai"

    @classmethod
    def build(cls, spec):
        try:
            from agents import Agent as OpenAIAgent
            from agents import ModelSettings
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
            "tools": [t.function for t in spec.tools],
        }
        if spec.model.parameters:
            agent_kwargs["model_settings"] = ModelSettings(**spec.model.parameters)
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
    def _resolve_model(model_spec):
        """Return an SDK model object only when per-agent client settings are needed."""
        if not model_spec.api_key:
            return model_spec.model

        try:
            from agents import OpenAIResponsesModel
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI API-key configuration requires the openai package. "
                "Install with: pip install 'agentframerelay[openai]'"
            ) from exc

        return OpenAIResponsesModel(
            model=model_spec.model,
            openai_client=AsyncOpenAI(api_key=model_spec.api_key),
        )

    @classmethod
    def capabilities(cls):
        return {"streaming": True, "memory": True, "human_in_loop": True,
                "durable_execution": False, "multi_agent": True}
