from app.retrieval.embedder import embed_batch, sparse_weights_to_qdrant_format
from app.retrieval.qdrant_store import hybrid_search
from app.retrieval.reranker import rerank
from app.guardrails.retrieval.relevance_grader import RelevanceGrader
from app.guardrails.retrieval.source_trust import filter_trusted

_grader = RelevanceGrader()


def retrieve(query: str, topic_filter: str | None = None, rerank_top_k: int = 5) -> list[dict]:
    emb = embed_batch([query])
    dense_vec = emb["dense_vecs"][0].tolist()
    indices, values = sparse_weights_to_qdrant_format(emb["sparse_weights"][0])

    hits = hybrid_search(dense_vec, indices, values, limit=15, topic_filter=topic_filter)
    candidates = [
        {
            "paper_id": h.payload["paper_id"],
            "title": h.payload["title"],
            "text": h.payload["text"],
            "section": h.payload.get("section", {}),
            "fusion_score": h.score,
        }
        for h in hits
    ]

    candidates = filter_trusted(candidates)
    candidates = rerank(query, candidates, top_k=rerank_top_k)
    candidates = _grader.filter_relevant(query, candidates)

    return candidates