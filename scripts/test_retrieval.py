import sys
sys.path.append(".")

from app.retrieval.embedder import embed_batch, sparse_weights_to_qdrant_format
from app.retrieval.qdrant_store import hybrid_search

query = "What are recent approaches for long-context transformer models?"

emb = embed_batch([query])
dense_vec = emb["dense_vecs"][0].tolist()
indices, values = sparse_weights_to_qdrant_format(emb["sparse_weights"][0])

results = hybrid_search(dense_vec, indices, values, limit=15)

for r in results:
    print(f"score={r.score:.4f} | paper={r.payload['paper_id']} | {r.payload['title'][:60]}")