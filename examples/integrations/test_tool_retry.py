from _bootstrap import ensure_local_package

ensure_local_package()

from agentframerelay import tool


attempts = 0


@tool(
    retries=3,
    retry_delay=1.0,
    backoff="constant",
)
def database_tool(query: str) -> str:
    global attempts

    attempts += 1

    print(f"🗄️ DATABASE EXECUTION #{attempts}")

    # Fail first two times
    if attempts < 3:
        raise ConnectionError("Database connection failed")

    return f"Query successful: {query}"


@database_tool.before
def before(context):
    print(
        f"🟡 BEFORE | attempt={context.attempt}"
    )


@database_tool.after
def after(context, result):
    print(
        f"🟢 AFTER | attempt={context.attempt} | result={result}"
    )


@database_tool.on_error
def on_error(context, error):
    print(
        f"🔴 ON_ERROR | "
        f"attempt={context.attempt} | "
        f"error={error}"
    )


print("\n========== RETRY SUCCESS TEST ==========\n")

try:
    result = database_tool.invoke(
        query="SELECT * FROM users"
    )

    print("\n🎉 FINAL RESULT:")
    print(result)

except Exception as e:
    print("\n❌ FINAL ERROR:")
    print(type(e).__name__)
    print(e)

print("\n========== TEST COMPLETE ==========")