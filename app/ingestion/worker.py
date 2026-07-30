from app.ingestion.pdf_parser import parse_pdf_to_markdown
from app.ingestion.chunker import chunk_markdown


def process_paper(pdf_path: str, result_queue):
    """Runs in its own OS process. Puts (status, payload) into result_queue.
    If this whole process crashes (native memory error), the parent script
    detects that via exitcode — it never needs this function to return."""
    try:
        markdown = parse_pdf_to_markdown(pdf_path)
        chunks = chunk_markdown(markdown)
        result_queue.put(("ok", chunks))
    except Exception as e:
        result_queue.put(("error", str(e)))