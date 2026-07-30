import asyncio
import json
import uuid
import streamlit as st
import websockets

WS_URL = "ws://localhost:8000/ws/chat"

st.set_page_config(page_title="arXiv Research Assistant", page_icon="📚", layout="wide")

# --- Session state setup ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": str, "citations": list}

# --- Sidebar ---
with st.sidebar:
    st.title("📚 Research Assistant")
    st.caption("Ask questions about your 100-paper knowledge base — or ask it to search arXiv for new papers.")

    topic_filter = st.selectbox(
        "Restrict to topic (optional)",
        options=["All topics", "Transformers and Deep Learning","Retrival-Augmented Generation (RAG)","Generative AI and Large Language Models (LLMs)","Agentic AI and Autonomous Systems"],  # ⚠️ replace with your real topic names
    )
    topic_filter = None if topic_filter == "All topics" else topic_filter

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

st.title("Ask your research question")

# --- Render chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("📎 Sources"):
                for c in msg["citations"]:
                    st.markdown(f"**{c['marker']}** {c['title']}  \n`{c['paper_id']}`")


async def stream_from_backend(query: str, session_id: str, topic_filter, placeholder):
    full_text = ""
    citations = []
    blocked_reason = None

    try:
        async with websockets.connect(WS_URL) as ws:
            await ws.send(json.dumps({"query": query, "session_id": session_id, "topic_filter": topic_filter}))

            while True:
                raw = await ws.recv()
                msg = json.loads(raw)

                if msg["type"] == "token":
                    full_text += msg["content"]
                    placeholder.markdown(full_text + "▌")
                elif msg["type"] == "end":
                    citations = msg.get("citations", [])
                    placeholder.markdown(full_text)
                    break
                elif msg["type"] == "blocked":
                    blocked_reason = msg.get("reason", "Query blocked by guardrails.")
                    break
    except websockets.exceptions.ConnectionClosedError as e:
        blocked_reason = f"WebSocket connection closed (Code: {e.code}). Please check backend logs."
    except Exception as e:
        blocked_reason = f"Failed to connect to backend: {e}"

    return full_text, citations, blocked_reason


# --- Chat input & execution ---
query = st.chat_input("Ask about your papers...")

if query:
    # 1. Save & display user message
    st.session_state.messages.append({"role": "user", "content": query, "citations": []})
    with st.chat_message("user"):
        st.markdown(query)

    # 2. Process assistant response ONLY when query exists
    with st.chat_message("assistant"):
        placeholder = st.empty()
        
        # Check for arXiv search keywords safely
        if any(kw in query.lower() for kw in ["arxiv", "latest papers", "new papers", "recent papers"]):
            placeholder.markdown("🔍 Fetching and processing new papers from arXiv — this can take 30-60s...")
        else:
            placeholder.markdown("Thinking...")

        full_text, citations, blocked_reason = asyncio.run(
            stream_from_backend(query, st.session_state.session_id, topic_filter, placeholder)
        )

        if blocked_reason:
            placeholder.warning(blocked_reason)
            st.session_state.messages.append({"role": "assistant", "content": blocked_reason, "citations": []})
        else:
            if citations:
                with st.expander("📎 Sources"):
                    for c in citations:
                        st.markdown(f"**{c['marker']}** {c['title']}  \n`{c['paper_id']}`")
            st.session_state.messages.append({"role": "assistant", "content": full_text, "citations": citations})