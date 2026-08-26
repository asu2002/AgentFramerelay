import asyncio
from pathlib import Path
import sys

from mcp import ClientSession
from mcp.client.stdio import (
    StdioServerParameters,
    stdio_client,
)


async def main():

    # =========================================================
    # 1. TELL THE CLIENT HOW TO START YOUR MCP SERVER
    # =========================================================

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[
            str(Path(__file__).with_name("test_mcp_server.py"))
        ],
    )


    # =========================================================
    # 2. CONNECT TO MCP SERVER
    # =========================================================

    async with stdio_client(
        server_params
    ) as (read, write):

        async with ClientSession(
            read,
            write,
        ) as session:

            # -------------------------------------------------
            # 3. INITIALIZE MCP CONNECTION
            # -------------------------------------------------

            await session.initialize()

            print("\n" + "=" * 60)
            print("CONNECTED TO MCP SERVER")
            print("=" * 60)


            # -------------------------------------------------
            # 4. LIST AVAILABLE TOOLS
            # -------------------------------------------------

            tools = await session.list_tools()

            print("\nAVAILABLE TOOLS:")

            for tool in tools.tools:

                print(f"\nName: {tool.name}")
                print(f"Description: {tool.description}")


            # -------------------------------------------------
            # 5. CALL ADD TOOL
            # -------------------------------------------------

            print("\n" + "-" * 60)
            print("CALLING ADD TOOL")
            print("-" * 60)

            result = await session.call_tool(
                "add",
                arguments={
                    "a": 25,
                    "b": 75,
                },
            )

            print("Result:")
            print(result)


            # -------------------------------------------------
            # 6. CALL MULTIPLY TOOL
            # -------------------------------------------------

            print("\n" + "-" * 60)
            print("CALLING MULTIPLY TOOL")
            print("-" * 60)

            result = await session.call_tool(
                "multiply",
                arguments={
                    "a": 10,
                    "b": 20,
                },
            )

            print("Result:")
            print(result)


            print("\n" + "=" * 60)
            print("MCP TEST SUCCESSFUL")
            print("=" * 60)


if __name__ == "__main__":

    asyncio.run(main())
