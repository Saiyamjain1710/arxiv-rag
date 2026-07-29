import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from app.agents.entrypoint import answer_query
from app.generation.context_assembler import assemble_context
from app.generation.citation_mapper import extract_used_citations
from app.generation.generator import stream_answer

app = FastAPI(title="arXiv RAG Assistant")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.websocket("/ws/chat")
async def chat_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            payload = json.loads(raw)
            query = payload["query"]
            session_id = payload.get("session_id", "default")
            topic_filter = payload.get("topic_filter")

            # Run heavy CPU/GPU sync operations in a separate threadpool
            # so the asyncio event loop remains responsive for WebSocket ping/pong
            result = await asyncio.to_thread(
                answer_query, 
                query, 
                session_id=session_id, 
                topic_filter=topic_filter
            )

            if result["blocked"]:
                await websocket.send_json({"type": "blocked", "reason": result["reason"]})
                continue

            chunks = result["retrieved_chunks"]
            if not chunks:
                await websocket.send_json({"type": "blocked", "reason": "No relevant information found in the knowledge base."})
                continue

            system_prompt, citation_lookup = assemble_context(chunks)

            await websocket.send_json({"type": "start"})

            full_text = ""
            async for token in stream_answer(query, system_prompt):
                full_text += token
                await websocket.send_json({"type": "token", "content": token})

            citations = extract_used_citations(full_text, citation_lookup)
            await websocket.send_json({"type": "end", "citations": citations})

    except WebSocketDisconnect:
        print("Client disconnected")