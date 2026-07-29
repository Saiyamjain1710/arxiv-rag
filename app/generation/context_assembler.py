SYSTEM_PROMPT = """You are an academic research assistant. Answer the user's question
using ONLY the numbered source passages provided below. For every factual claim,
cite the source number in square brackets, e.g. [1], or [1][3] if multiple sources
support it. If the passages don't contain enough information to answer, say so
honestly instead of guessing.

Sources:
{context_block}
"""


def assemble_context(chunks: list[dict]) -> tuple[str, dict]:
    """Returns (system_prompt_with_context, citation_lookup).
    citation_lookup maps "1" -> {paper_id, title, text} etc., so the citation
    mapper (next file) can resolve [1] in the model's output back to real metadata."""
    context_lines = []
    citation_lookup = {}

    for i, chunk in enumerate(chunks, start=1):
        context_lines.append(f"[{i}] (from \"{chunk['title']}\"): {chunk['text']}")
        citation_lookup[str(i)] = {
            "paper_id": chunk["paper_id"],
            "title": chunk["title"],
            "text": chunk["text"],
        }

    context_block = "\n\n".join(context_lines)
    system_prompt = SYSTEM_PROMPT.format(context_block=context_block)
    return system_prompt, citation_lookup