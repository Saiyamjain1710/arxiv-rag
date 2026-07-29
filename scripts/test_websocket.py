import asyncio
import json
import websockets

async def main():
    # Matches endpoint in main.py: /ws/chat
    uri = "ws://127.0.0.1:8000/ws/chat"  
    
    print("Connecting to server...")
    async with websockets.connect(
        uri, 
        ping_interval=None, 
        ping_timeout=None
    ) as ws:
        print("Connected! Sending query...")
        
        # Send JSON payload matching main.py expectations
        payload = {
            "query": "What is transformers architecture?",
            "session_id": "test_session"
        }
        await ws.send(json.dumps(payload))
        
        while True:
            try:
                raw = await ws.recv()
                msg = json.loads(raw)
                msg_type = msg.get("type")

                if msg_type == "start":
                    print("\n--- Streaming Response ---")
                elif msg_type == "token":
                    print(msg.get("content", ""), end="", flush=True)
                elif msg_type == "end":
                    print("\n--- End of Response ---")
                    print("Citations:", msg.get("citations", []))
                    break
                elif msg_type == "blocked":
                    print(f"\nRequest blocked: {msg.get('reason')}")
                    break

            except websockets.exceptions.ConnectionClosedOK:
                print("Connection closed cleanly by server.")
                break
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"Connection closed with error: {e}")
                break

if __name__ == "__main__":
    asyncio.run(main())