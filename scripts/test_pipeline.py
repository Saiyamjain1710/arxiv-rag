import sys
sys.path.append(".")
from app.retrieval.pipeline import retrieve

query = "What are recent approaches for long-context transformer models?"
results = retrieve(query)

print(f"\n{len(results)} chunks passed all guardrails:\n")
for r in results:
    print(f"rerank_score={r['rerank_score']:.4f} | {r['paper_id']} | {r['title'][:60]}")