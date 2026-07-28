from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from app.retrieval.qdrant_store import hybrid_search
from app.retrieval.embedder import embed_batch, sparse_weights_to_qdrant_format
from app.retrieval.pipeline import retrieve
from app.agents.arxiv_fetcher import fetch_and_index
from app.config import settings

EXPLICIT_FETCH_KEYWORDS = [
    "latest papers", "search arxiv", "new papers", "recent papers on arxiv",
    "look up new research", "find new papers",
]


class RAGState(TypedDict):
    query: str
    session_id: str
    topic_filter: Optional[str]
    confidence_score: float
    retrieved_chunks: list


def is_explicit_fetch_request(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in EXPLICIT_FETCH_KEYWORDS)


def score_node(state: RAGState) -> RAGState:
    """Confidence Scorer: a cheap top-1 hybrid search score, BEFORE reranking,
    just to decide which branch to take."""
    emb = embed_batch([state["query"]])
    dense_vec = emb["dense_vecs"][0].tolist()
    indices, values = sparse_weights_to_qdrant_format(emb["sparse_weights"][0])
    hits = hybrid_search(dense_vec, indices, values, limit=1, topic_filter=state.get("topic_filter"))
    score = hits[0].score if hits else 0.0
    print(f"  [confidence_scorer] top score = {score:.4f}")
    return {**state, "confidence_score": score}


def route_decision(state: RAGState) -> str:
    if is_explicit_fetch_request(state["query"]):
        return "arxiv"
    score = state["confidence_score"]
    if score >= settings.confidence_tau_high:
        return "local"
    elif score >= settings.confidence_tau_low:
        return "hybrid"
    else:
        return "arxiv"


def local_node(state: RAGState) -> RAGState:
    chunks = retrieve(state["query"], topic_filter=state.get("topic_filter"))
    return {**state, "retrieved_chunks": chunks}


def arxiv_node(state: RAGState) -> RAGState:
    new_ids = fetch_and_index(state["query"], max_results=settings.arxiv_max_fetch)
    chunks = retrieve(state["query"], topic_filter=state.get("topic_filter"), extra_trusted_ids=set(new_ids))
    return {**state, "retrieved_chunks": chunks}


def hybrid_node(state: RAGState) -> RAGState:
    local_chunks = retrieve(state["query"], topic_filter=state.get("topic_filter"))
    new_ids = fetch_and_index(state["query"], max_results=settings.arxiv_max_fetch)
    arxiv_chunks = retrieve(state["query"], topic_filter=state.get("topic_filter"), extra_trusted_ids=set(new_ids))

    combined = {c["text"]: c for c in local_chunks + arxiv_chunks}
    merged = sorted(combined.values(), key=lambda c: c["rerank_score"], reverse=True)
    return {**state, "retrieved_chunks": merged[:5]}


def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("score", score_node)
    graph.add_node("local", local_node)
    graph.add_node("arxiv", arxiv_node)
    graph.add_node("hybrid", hybrid_node)

    graph.set_entry_point("score")
    graph.add_conditional_edges("score", route_decision, {"local": "local", "hybrid": "hybrid", "arxiv": "arxiv"})
    graph.add_edge("local", END)
    graph.add_edge("hybrid", END)
    graph.add_edge("arxiv", END)

    return graph.compile()


rag_graph = build_graph()