import sys
sys.path.append(".")
from app.agents.entrypoint import answer_query

print("=== Test 1: should hit local retrieval (about your existing papers) ===")
result = answer_query("What are recent approaches for long-context transformer models?", session_id="test1")
print("Blocked:", result["blocked"])
for c in result.get("retrieved_chunks", []):
    print(f"  {c['paper_id']} | {c['title'][:60]} | score={c.get('rerank_score', 0):.4f}")

print("\n=== Test 2: explicit fetch request (should go to arXiv agent) ===")
result = answer_query("Search arxiv for the latest papers on diffusion models", session_id="test2")
print("Blocked:", result["blocked"])
for c in result.get("retrieved_chunks", []):
    print(f"  {c['paper_id']} | {c['title'][:60]} | score={c.get('rerank_score', 0):.4f}")