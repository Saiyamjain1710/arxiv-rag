from docling.document_converter import DocumentConverter

_converter = DocumentConverter()


def parse_pdf_to_markdown(pdf_path: str) -> str:
    """Parses a PDF into structured Markdown (preserves headers, so our
    chunker can split on # / ## boundaries in the next step)."""
    result = _converter.convert(pdf_path)
    return result.document.export_to_markdown()