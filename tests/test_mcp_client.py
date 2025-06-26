#%%
import asyncio
import nest_asyncio
import os
from dotenv import load_dotenv

nest_asyncio.apply()  # Only needed if running in a Jupyter notebook
load_dotenv()
TRANSPORT = os.getenv("TRANSPORT", "http")

async def wait_for_server(url, timeout=3):
    import httpx
    for _ in range(timeout * 2):
        try:
            async with httpx.AsyncClient() as client:
                await client.get(url)
            print(f"Server is up at {url}")
            return True
        except Exception:
            await asyncio.sleep(0.5)
    print(f"ERROR: Server did not start in time at {url}")
    return False

async def list_tools(session):
    tools_result = await session.list_tools()
    print("Available tools:")
    for tool in tools_result.tools:
        print(f"  - {tool.name}: {tool.description}")
    return tools_result

async def call_route_request(session):
    result = await session.call_tool(
        "route_request",
        arguments={"messages": [{"role": "user", "content": "Hello!"}]}
    )
    print("Tool result:", result.content[0].text)
    return result

async def run_session(session):
    # Initialize the connection
    await session.initialize()

    await list_tools(session)
    await call_route_request(session)

async def run_sse():
    from mcp.client.sse import sse_client
    from mcp import ClientSession
    # Poll root endpoint for readiness, not /sse
    if not await wait_for_server("http://localhost:8050/", timeout=10):
        return  # Exit early if server is not up
    async with sse_client("http://localhost:8050/sse") as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await run_session(session)

async def run_stdio():
    from mcp.client.stdio import stdio_client, StdioServerParameters
    from mcp import ClientSession
    # Get absolute path to server.py relative to this test file's location
    server_py_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../smart-intent-router-server/src/mcp_server/server.py'))
    server_params = StdioServerParameters(
        command="python",
        args=[server_py_path],
    )

    # Connect to the server
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await run_session(session)

async def run_http():
    import httpx
    # Wait for server readiness
    if not await wait_for_server("http://localhost:8050/route_request", timeout=10):
        return
    async with httpx.AsyncClient(timeout=None) as client:
        # Single-response test
        print("\n--- HTTP /route_request (single-response) ---")
        resp = await client.post("http://localhost:8050/route_request", json={
            "messages": [{"role": "user", "content": "Hello!"}]
        })
        print("Response:", resp.json())

        # Streaming test
        print("\n--- HTTP /route_request_stream (streaming) ---")
        async with client.stream("POST", "http://localhost:8050/route_request_stream", json={
            "messages": [{"role": "user", "content": "Hello!"}]
        }) as stream_resp:
            print("Streamed response:", end=" ")
            async for chunk in stream_resp.aiter_text():
                print(chunk, end="", flush=True)
            print()

async def main():
    if TRANSPORT == "sse":
        await run_sse()
    elif TRANSPORT == "http":
        await run_http()
    else:
        await run_stdio()

if __name__ == "__main__":
    asyncio.run(main())

# %%
