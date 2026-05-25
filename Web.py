import aiohttp
import pickle
from aiohttp import web
import asyncio
import os
from DataStructure import query_packet
cloud_server=os.getenv("RAG_CLOUD_SERVER_URL", "ws://localhost:20000/ws")
async def edge_to_cloud(query_packet):
    """
    Send a serialized query packet from the edge to the cloud server.

    This demo helper is not part of the main Lever reproduction pipeline.
    """
    data=pickle.dumps(query_packet)
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(cloud_server) as ws:
            # Send the initial payload.
            await ws.send_bytes(data)

            # Keep receiving server messages.
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    print(f"Received server response: {msg.data}")
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    print("Connection closed cleanly.")
                    break
connected_clients = set()
# Websocket server entry point used for demo purposes.
async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    connected_clients.add(ws)

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                q_packet = pickle.loads(msg.data)
                for client in connected_clients:
                    await client.send_str(f"Received: {msg.data}")
    finally:
        connected_clients.remove(ws)
    return ws
if __name__=='__main__':
    raise SystemExit("This file contains demo websocket helpers only.")







