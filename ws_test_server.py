import asyncio
import websockets
import json

async def handler(websocket):
    async for message in websocket:
        data = json.loads(message)
        print("Received from Python:", data["gesture"])

async def main():
    server = await websockets.serve(handler, "localhost", 8765)
    print("WebSocket server running on ws://localhost:8765")
    await server.wait_closed()  # Keep server running

# Run the server
asyncio.run(main())
