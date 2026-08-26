import asyncio
import keyword
import re

from ..runtime import RuntimeAdapter, RuntimeResult
from ..tool import Tool


class GoogleADKAdapter(RuntimeAdapter):
    name = "google_adk"

    @classmethod
    def tool(cls, tool: Tool):
        # Google ADK accepts typed Python callables as function tools.
        return tool.function

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
            cls.tool(Tool(t.function, name=t.name, description=t.description))
            for t in spec.tools
        ]

        return ADKAgent(
            name=cls._native_name(spec.name),
            model=model,
            instruction=spec.instructions or f"You are {spec.name}.",
            tools=tools,
        )

    @staticmethod
    def _resolve_model(model_spec):
        model = model_spec.model if hasattr(model_spec, "model") else str(model_spec)
        if not getattr(model_spec, "api_key", None):
            return model

        try:
            from google.adk.models.google_llm import Gemini
        except ImportError as exc:
            raise ImportError(
                "Google ADK support requires google-adk. "
                "Install with: pip install 'agentframerelay[google-adk]'"
            ) from exc
        return Gemini(model=model, client_kwargs={"api_key": model_spec.api_key})

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
