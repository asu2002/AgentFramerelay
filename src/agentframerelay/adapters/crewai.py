import inspect
from functools import wraps

from crewai import Agent as CrewAgent
from crewai import Task, Crew
from crewai.tools import tool as crew_tool

from agentframerelay.runtime import RuntimeResult
from agentframerelay.tool import Tool


class CrewAIAdapter:

    name = "crewai"

    # =========================================================
    # PROVIDER-SPECIFIC COMPATIBILITY
    # =========================================================

    @classmethod
    def _apply_provider_compatibility(cls, model_spec):
        """
        Apply compatibility fixes only when required.

        This keeps the adapter provider-neutral and prevents
        Groq-specific behavior from affecting OpenAI, Anthropic,
        Gemini, Bedrock, etc.
        """

        provider = (
            model_spec.provider or ""
        ).lower()

        if provider != "groq":
            return

        # -----------------------------------------------------
        # Groq compatibility workaround
        #
        # Some CrewAI/LiteLLM combinations may inject
        # `cache_breakpoint` into messages.
        # Groq may reject this field.
        # -----------------------------------------------------

        def remove_cache_breakpoint(message):
            if isinstance(message, dict):
                message.pop("cache_breakpoint", None)

            return message

        try:
            import crewai.llms.cache as cache

            if hasattr(cache, "mark_cache_breakpoint"):
                cache.mark_cache_breakpoint = (
                    remove_cache_breakpoint
                )

        except ImportError:
            pass

        try:
            import crewai.agents.crew_agent_executor as executor

            if hasattr(
                executor,
                "mark_cache_breakpoint"
            ):
                executor.mark_cache_breakpoint = (
                    remove_cache_breakpoint
                )

        except ImportError:
            pass


    # =========================================================
    # MODEL RESOLVER
    # =========================================================

    @classmethod
    def _resolve_model(cls, model_spec):

        try:
            from crewai import LLM

        except ImportError as exc:
            raise ImportError(
                "CrewAI support requires crewai.\n"
                "Install with: pip install crewai"
            ) from exc

        provider = (
            model_spec.provider or ""
        ).strip()

        model = model_spec.model

        # -----------------------------------------------------
        # Build provider/model name
        # -----------------------------------------------------

        if provider:
            model_name = f"{provider}/{model}"
        else:
            model_name = model

        # -----------------------------------------------------
        # Copy neutral parameters
        # -----------------------------------------------------

        parameters = dict(
            model_spec.parameters or {}
        )

        # -----------------------------------------------------
        # Build CrewAI LLM
        # -----------------------------------------------------

        llm_kwargs = {
            "model": model_name,
            **parameters,
        }

        # API key is optional.
        # This allows providers using environment variables,
        # IAM credentials, AWS credentials, etc.
        if getattr(model_spec, "api_key", None):
            llm_kwargs["api_key"] = (
                model_spec.api_key
            )

        return LLM(**llm_kwargs)


    # =========================================================
    # TOOL CONVERTER
    # =========================================================

    @classmethod
    def tool(cls, relay_tool: Tool):
        """
        Convert an AgentFrameRelay Tool
        into a native CrewAI tool.
        """

        original_function = relay_tool.function

        @crew_tool(relay_tool.name)
        @wraps(original_function)
        def crewai_wrapped_tool(*args, **kwargs):

            return original_function(
                *args,
                **kwargs
            )

        # Preserve the original function signature so CrewAI
        # can detect arguments and their types.
        crewai_wrapped_tool.__signature__ = (
            inspect.signature(original_function)
        )

        return crewai_wrapped_tool


    # =========================================================
    # BUILD CREWAI AGENT
    # =========================================================

    @classmethod
    def build(cls, spec):

        # Apply provider-specific compatibility only when needed.
        cls._apply_provider_compatibility(
            spec.model
        )

        # Convert AgentFrameRelay tools -> CrewAI tools.
        tools = [
            cls.tool(tool)
            for tool in spec.tools
        ]

        # Convert neutral ModelSpec -> CrewAI LLM.
        llm = cls._resolve_model(
            spec.model
        )

        return CrewAgent(
            role=(
                getattr(spec, "role", None)
                or spec.name
            ),

            goal=(
                getattr(spec, "goal", None)
                or "Complete the assigned task successfully."
            ),

            backstory=(
                getattr(spec, "instructions", None)
                or f"You are {spec.name}."
            ),

            tools=tools,

            llm=llm,

            verbose=True,
        )


    # =========================================================
    # RUN CREWAI AGENT
    # =========================================================

    @classmethod
    def run(cls, native_agent, input, **kwargs):

        task = Task(
            description=input,

            expected_output=kwargs.pop(
                "expected_output",
                "Provide a complete and accurate answer."
            ),

            agent=native_agent,
        )

        crew = Crew(
            agents=[native_agent],

            tasks=[task],

            verbose=kwargs.pop(
                "verbose",
                True
            ),
        )

        return RuntimeResult(
            output=crew.kickoff(),
            runtime=cls.name,
        )


    # =========================================================
    # CAPABILITIES
    # =========================================================

    @classmethod
    def capabilities(cls):

        return {
            "streaming": False,
            "memory": True,
            "human_in_loop": True,
            "durable_execution": True,
            "multi_agent": True,
        }
