"""Sanity check: list the fake server's tools over stdio with the official MCP client."""
import asyncio, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(command=sys.executable, args=["fake_server.py"])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            for t in (await s.list_tools()).tools:
                print(t.name, "::", t.description.replace("\n", " | "))
asyncio.run(main())
