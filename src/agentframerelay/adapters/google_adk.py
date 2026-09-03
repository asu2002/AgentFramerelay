import asyncio
import keyword
import re
from typing import ClassVar

from ..runtime import RuntimeAdapter, RuntimeResult
from ..tool import Tool


class GoogleADKAdapter(RuntimeAdapter):
    name = "google_adk"
    _GOOGLE_PROVIDERS: ClassVar[frozenset[str]] = frozenset({"google", "google_ai", "gemini"})

    @staticmethod
    def _groq_model_class(lite_llm_class):
        """Return an ADK LiteLLM model that omits Groq-unsupported history."""

        class GroqLiteLlm(lite_llm_class):
            async def generate_content_async(self, llm_request, stream=False):
                # ADK maps prior ``Part(thought=True)`` values to the
                # OpenAI-compatible ``reasoning_content`` message field.
                # Groq rejects that field, so retain the conversation and tool
                # calls while excluding only the replayed reasoning history.
                sanitized_request = llm_request.model_copy(deep=True)
                for content in sanitized_request.contents:
                    if content.parts:
                        content.parts = [part for part in content.parts if not part.thought]
                async for response in super().generate_content_async(
                    sanitized_request, stream=stream
                ):
                    yield response

        return GroqLiteLlm

    @classmethod
    def tool(cls, tool: Tool):
        # Google ADK accepts typed Python callables as function tools.
        return tool.adapter_callable()

    @classmethod
    def build(cls, spec):
        try:
            from google.adk.agents import Agent as ADKAgent
        except ImportError as exc:
            raise ImportError(
                "Google ADK support requires google-adk. "
                "Install with: pip install 'agentframerelay[google-adk]'"
            ) from exc

        if not spec.model:
            raise ValueError("A model is required for the Google ADK adapter.")

        model = cls._resolve_model(spec.model)
        tools = [
            cls.tool(Tool.from_spec(t))
            for t in spec.tools
        ]

        return ADKAgent(
            name=cls._native_name(spec.name),
            model=model,
            instruction=spec.instructions or f"You are {spec.name}.",
            tools=tools,
        )

    @classmethod
    def _resolve_model(cls, model_spec):
        provider = (getattr(model_spec, "provider", "") or "").strip().lower()
        model = model_spec.model if hasattr(model_spec, "model") else str(model_spec)
        parameters = dict(getattr(model_spec, "parameters", {}) or {})

        if provider in cls._GOOGLE_PROVIDERS:
            # A bare model name preserves Google ADK's standard environment
            # credential resolution. Configure Gemini only when needed.
            if not getattr(model_spec, "api_key", None) and not parameters:
                return model
            allowed = {
                "base_url", "client_kwargs", "speech_config",
                "use_interactions_api", "retry_options",
            }
            unsupported = sorted(set(parameters) - allowed)
            if unsupported:
                raise ValueError(
                    "Google ADK adapter does not support Gemini configuration "
                    f"parameters for model '{model}': {', '.join(unsupported)}."
                )
            try:
                from google.adk.models.google_llm import Gemini
            except ImportError as exc:
                raise ImportError(
                    "Google ADK support requires google-adk. "
                    "Install with: pip install 'agentframerelay[google-adk]'"
                ) from exc
            client_kwargs = dict(parameters.pop("client_kwargs", {}) or {})
            if getattr(model_spec, "api_key", None):
                client_kwargs["api_key"] = model_spec.api_key
            return Gemini(model=model, client_kwargs=client_kwargs or None, **parameters)

        try:
            import litellm
            from google.adk.models.lite_llm import LiteLlm
        except ImportError as exc:
            raise ImportError(
                "Google ADK non-Google provider support requires LiteLLM. "
                "Install with: pip install 'agentframerelay[google-adk,litellm]'"
            ) from exc
        if provider not in litellm.provider_list:
            raise ValueError(
                "Google ADK adapter does not support "
                f"provider '{model_spec.provider}' for model '{model}'."
            )
        if getattr(model_spec, "api_key", None):
            parameters["api_key"] = model_spec.api_key
        model_class = cls._groq_model_class(LiteLlm) if provider == "groq" else LiteLlm
        return model_class(model=f"{provider}/{model}", **parameters)

    @staticmethod
    def _native_name(name: str) -> str:
        """Convert a framework-neutral display name into an ADK node name."""
        native_name = re.sub(r"\W", "_", name)
        if not native_name or native_name[0].isdigit() or keyword.iskeyword(native_name):
            native_name = f"agent_{native_name}"
        return native_name

    @classmethod
    def run(cls, native_agent, input, **kwargs):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(cls._run(native_agent, input, **kwargs))
        raise RuntimeError(
            "GoogleADKAdapter.run() cannot run inside an active event loop. "
            "Use the native Google ADK Runner in asynchronous applications."
        )

    @classmethod
    async def _run(cls, native_agent, input, **kwargs) -> RuntimeResult:
        try:
            from google.adk.runners import Runner
            from google.adk.sessions import InMemorySessionService
            from google.genai import types
        except ImportError as exc:
            raise ImportError(
                "Google ADK support requires google-adk. "
                "Install with: pip install 'agentframerelay[google-adk]'"
            ) from exc

        app_name = kwargs.pop("app_name", native_agent.name)
        user_id = kwargs.pop("user_id", "agentframerelay-user")
        session_id = kwargs.pop("session_id", None)
        state = kwargs.pop("state", None)
        session_service = kwargs.pop("session_service", None) or InMemorySessionService()
        run_config = kwargs.pop("run_config", None)
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported Google ADK run options: {names}")

        if session_id:
            session = await session_service.get_session(
                app_name=app_name, user_id=user_id, session_id=session_id
            )
        else:
            session = None
        if session is None:
            session = await session_service.create_session(
                app_name=app_name, user_id=user_id, session_id=session_id, state=state
            )

        message = input if isinstance(input, types.Content) else types.Content(
            role="user", parts=[types.Part(text=str(input))]
        )
        runner = Runner(
            app_name=app_name, agent=native_agent, session_service=session_service
        )
        events = []
        final_event = None
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=message,
            run_config=run_config,
        ):
            events.append(event)
            if event.is_final_response():
                final_event = event

        return RuntimeResult(
            output=final_event or (events[-1] if events else None),
            runtime=cls.name,
            metadata={
                "app_name": app_name,
                "user_id": user_id,
                "session_id": session.id,
                "events": len(events),
            },
        )

    @classmethod
    def capabilities(cls):
        return {
            "streaming": False,
            "memory": True,
            "human_in_loop": False,
            "durable_execution": True,
            "multi_agent": True,
        }
