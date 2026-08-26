from mcp.server.fastmcp import FastMCP

from _bootstrap import ensure_local_package

ensure_local_package()

from agentframerelay import tool


# ============================================================
# CREATE MCP SERVER
# ============================================================

mcp = FastMCP("AgentFrameRelay MCP Server")


# ============================================================
# CREATE TOOL USING AGENTFRAMERELAY
# ============================================================

@tool
def add(a: int, b: int) -> int:
    """Add two integers together."""

    print(f"[AGENTFRAMERELAY TOOL CALLED] add({a}, {b})")

    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers together."""

    print(
        f"[AGENTFRAMERELAY TOOL CALLED] "
        f"multiply({a}, {b})"
    )

    return a * b


# ============================================================
# REGISTER AGENTFRAMERELAY TOOLS WITH MCP
# ============================================================

add.register_mcp(mcp)
multiply.register_mcp(mcp)


# ============================================================
# RUN MCP SERVER
# ============================================================

if __name__ == "__main__":
    mcp.run()
