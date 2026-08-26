from agentframerelay.tool import Tool


class MCPAdapter:
    """
    Converts AgentFrameRelay tools into MCP-compatible tools.
    """

    @classmethod
    def tool(cls, relay_tool: Tool):
        """
        MCP/FastMCP can work directly with a typed Python function.

        We return the original function so its:
        - name
        - type annotations
        - docstring
        remain available to the MCP framework.
        """

        return relay_tool.function


    @classmethod
    def register(cls, relay_tool: Tool, server):
        """
        Register an AgentFrameRelay tool directly with an MCP server.
        """

        return server.tool(
            name=relay_tool.name,
            description=relay_tool.description,
        )(relay_tool.function)