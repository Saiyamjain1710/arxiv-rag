import re

CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def extract_used_citations(generated_text: str, citation_lookup: dict) -> list[dict]:
    """Scans the model's streamed-so-far text for [N] markers and returns the
    unique source metadata actually cited, in order of first appearance."""
    seen = set()
    used = []
    for match in CITATION_PATTERN.finditer(generated_text):
        num = match.group(1)
        if num in citation_lookup and num not in seen:
            seen.add(num)
            used.append({"marker": f"[{num}]", **citation_lookup[num]})
    return used