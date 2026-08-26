from _bootstrap import ensure_local_package

ensure_local_package()

from agentframerelay import tool
import os
from crewai import Agent, Task, Crew, LLM


# ============================================================
# 1. CREWAI + GROQ COMPATIBILITY WORKAROUND
# ============================================================

def apply_groq_compatibility():
    """
    Temporary workaround for CrewAI + Groq.

    Some CrewAI versions may add `cache_breakpoint`
    to messages, which Groq rejects.
    """

    def remove_cache_breakpoint(message):

        if isinstance(message, dict):
            message.pop("cache_breakpoint", None)

        return message

    # Patch CrewAI cache module
    try:
        import crewai.llms.cache as cache

        if hasattr(cache, "mark_cache_breakpoint"):
            cache.mark_cache_breakpoint = remove_cache_breakpoint

    except (ImportError, AttributeError):
        pass

    # Patch executor in case it imported the function directly
    try:
        import crewai.agents.crew_agent_executor as executor

        if hasattr(executor, "mark_cache_breakpoint"):
            executor.mark_cache_breakpoint = remove_cache_breakpoint

    except (ImportError, AttributeError):
        pass


# Apply compatibility before creating/running the Crew
apply_groq_compatibility()


# ============================================================
# 2. CREATE TOOL USING AGENTFRAMERELAY
# ============================================================

@tool
def add(a: int, b: int) -> int:
    """Add two integers together."""

    print("\n" + "=" * 60)
    print("🔥 AGENTFRAMERELAY TOOL WAS EXECUTED")
    print(f"add({a}, {b})")
    print("=" * 60 + "\n")

    return a + b


# ============================================================
# 3. CONVERT AGENTFRAMERELAY TOOL -> CREWAI TOOL
# ============================================================

crewai_add_tool = add.to_crewai()


# ============================================================
# 4. CREATE NATIVE CREWAI LLM
# ============================================================

llm = LLM(
    model="groq/openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)


# ============================================================
# 5. CREATE NATIVE CREWAI AGENT
# ============================================================

math_agent = Agent(

    role="Math Assistant",

    goal=(
        "Solve mathematical problems using the available tools."
    ),

    backstory=(
        "You are a mathematical assistant. "
        "You must use the provided add tool when asked "
        "to perform addition. Do not calculate manually."
    ),

    tools=[
        crewai_add_tool
    ],

    llm=llm,

    verbose=True,
)


# ============================================================
# 6. CREATE NATIVE CREWAI TASK
# ============================================================

task = Task(

    description="""
Calculate 25 + 75.

IMPORTANT:
You MUST use the add tool.
Do not calculate the answer yourself.
""",

    expected_output=(
        "The result of adding 25 and 75."
    ),

    agent=math_agent,
)


# ============================================================
# 7. CREATE NATIVE CREWAI CREW
# ============================================================

crew = Crew(

    agents=[
        math_agent
    ],

    tasks=[
        task
    ],

    verbose=True,
)


# ============================================================
# 8. RUN CREWAI
# ============================================================

result = crew.kickoff()


# ============================================================
# 9. PRINT RESULT
# ============================================================

print("\n" + "=" * 60)
print("FINAL RESULT")
print("=" * 60)

print(result.raw)
