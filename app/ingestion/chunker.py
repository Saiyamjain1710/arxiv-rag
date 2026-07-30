from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import tiktoken

_encoder = tiktoken.get_encoding("cl100k_base")

HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
]

_header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)

MAX_TOKENS = 800
MIN_TOKENS_SOFT_TARGET = 600


def _token_len(text: str) -> int:
    return len(_encoder.encode(text, disallowed_special=()))


def chunk_markdown(markdown_text: str) -> list[dict]:
    """Splits on headers first (Related Work, Experiments, etc.), then
    applies a soft ~600-800 token cap with paragraph-boundary fallback for
    any section that's still too long (common in CS/ML papers)."""
    header_sections = _header_splitter.split_text(markdown_text)

    # paragraph-aware fallback splitter for oversized sections
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_TOKENS * 4,  # approx chars-per-token=4, refined below
        chunk_overlap=0,
        separators=["\n\n", "\n", ". ", " "],
    )

    final_chunks = []
    for section in header_sections:
        text = section.page_content
        if _token_len(text) <= MAX_TOKENS:
            final_chunks.append({"text": text, "metadata": section.metadata})
        else:
            sub_texts = fallback_splitter.split_text(text)
            for sub in sub_texts:
                final_chunks.append({"text": sub, "metadata": section.metadata})

    return final_chunks