from _bootstrap import ensure_local_package

ensure_local_package()
import os
from agentframerelay import Agent, tool


# @tool(
#     retries=3,
#     retry_delay=1,
#     backoff="exponential",
# )
# def failing_database_tool(query: str) -> str:

#     print("🔥 TOOL CALLED")

#     raise RuntimeError(
#         "Database connection failed"
#     )


# @failing_database_tool.before
# def before_hook(context):

#     print(
#         f"🟡 BEFORE "
#         f"| attempt={context.attempt}"
#         f" | tool={context.tool_name}"
#     )


# @failing_database_tool.after
# def after_hook(context, result):

#     print(
#         f"🟢 AFTER "
#         f"| attempt={context.attempt}"
#         f" | result={result}"
#     )


# @failing_database_tool.on_error
# def error_hook(context, error):

#     print(
#         f"🔴 ERROR "
#         f"| attempt={context.attempt}"
#         f" | error={error}"
#     )


# agent = Agent(
#     name="retry_test_agent",

#     instructions="""
# You are testing a database tool.

# Always call the failing_database_tool when asked
# to query the database.

# Do not attempt to answer without calling the tool.
# """,

#     runtime="langchain",

#     model={
#            "provider": "google",
#            "model": os.getenv("GOOGLE_ADK_MODEL", "gemini-3.6-flash"),
#            "api_key": os.getenv("GOOGLE_API_KEY")
#        },

#     tools=[
#         failing_database_tool
#     ],
# )


# print("\nSTARTING AGENT...\n")

# try:

#     result = agent.run(
#         "Query the database and tell me how many users exist."
#     )

#     print("\nFINAL RESULT:")
#     print(result)

# except Exception as e:

#     print("\nFINAL EXCEPTION:")
#     print(type(e).__name__)
#     print(e)



# ============================================================
# TOOL STATE
# ============================================================

attempts = 0


# ============================================================
# TOOL
# ============================================================

@tool(
    retries=2,
    retry_delay=1.0,
    backoff="constant",
)
def unstable_database(query: str) -> str:
    """
    Simulates a database that fails twice and succeeds
    on the third attempt.
    """

    global attempts
    attempts += 1

    print(f"\n🔧 TOOL EXECUTION #{attempts}")
    print(f"   Query: {query}")

    if attempts <= 2:
        print("❌ Database failed")
        raise RuntimeError("Temporary database connection failure")

    print("✅ Database succeeded")

    return "Database query successful. Found 42 users."


# ============================================================
# HOOKS
# ============================================================

@unstable_database.before
def before_hook(context):

    print(
        f"🟡 BEFORE HOOK"
        f" | attempt={context.attempt}"
        f" | tool={context.tool_name}"
    )


@unstable_database.after
def after_hook(context, result):

    print(
        f"🟢 AFTER HOOK"
        f" | attempt={context.attempt}"
    )

    print(f"   Result: {result}")


@unstable_database.on_error
def error_hook(context, error):

    print(
        f"🔴 ERROR HOOK"
        f" | attempt={context.attempt}"
    )

    print(f"   Error: {error}")


# ============================================================
# AGENT
# ============================================================

agent = Agent(
    name="database_agent",

    instructions="""
You are a database assistant.

When the user asks for database information,
you MUST call the unstable_database tool.

Do not answer from your own knowledge.

After the tool succeeds, tell the user the result.
""",

    runtime="litellm",

    model={
        "provider": "google",
        "model": os.getenv("GOOGLE_ADK_MODEL", "gemini-3.7-flash"),
        "api_key": os.getenv("GOOGLE_API_KEY")
    },

    tools=[
        unstable_database
    ],
)


# ============================================================
# RUN AGENT
# ============================================================

print("\n" + "=" * 70)
print("STARTING AGENT")
print("=" * 70)

try:

    result = agent.run(
        "Please query the database and tell me how many users there are."
    )

    print("\n" + "=" * 70)
    print("FINAL AGENT RESULT")
    print("=" * 70)

    print(result)

except Exception as e:

    print("\n" + "=" * 70)
    print("AGENT FAILED")
    print("=" * 70)

    print(type(e).__name__)
    print(e)


print("\n" + "=" * 70)
print(f"TOTAL TOOL EXECUTIONS: {attempts}")
print("=" * 70)