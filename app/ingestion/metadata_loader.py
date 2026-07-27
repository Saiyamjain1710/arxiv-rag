import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

# Define project root relative to this file: app/ingestion/metadata_loader.py -> root is 2 levels up
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_META_DIR = PROJECT_ROOT / "data" / "metadata"  # Adjust "data/metadata" or "metadata" based on your layout

FIELD_MAPPING = {
    "paper_id": "arxiv_id",          # Using arxiv_id as internal paper identifier
    "title": "title",
    "authors": "authors",            # List of dicts e.g. [{"name": "..."}, ...]
    "topic": "primary_category",     # Maps to arXiv primary category (e.g., cs.CL)
    "abstract": "abstract",
    "pdf_filename": "arxiv_id",      # Mapped to arxiv_id to reconstruct filename
}


@dataclass
class PaperMetadata:
    paper_id: str
    title: str
    authors: list
    topic: str
    abstract: str
    pdf_filename: str
    citation_count: Optional[int] = None
    publication_date: Optional[str] = None


def load_metadata(meta_dir: Optional[str] = None) -> List[PaperMetadata]:
    """
    Loads metadata from either:
    1. A single directory of individual JSON files (e.g., metadata/2005.11401.json).
    2. A single combined JSON file (list or dict-of-dicts).
    """
    path = Path(meta_dir) if meta_dir else DEFAULT_META_DIR
    records = []

    if path.is_dir():
        # Read all individual .json files saved by the download script
        for json_file in path.glob("*.json"):
            with open(json_file, "r", encoding="utf-8") as f:
                records.append(json.load(f))
    elif path.is_file():
        # Fallback for a single aggregate JSON file
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            records = raw if isinstance(raw, list) else list(raw.values())
    else:
        raise FileNotFoundError(
            f"Path not found: {path.resolve()}\n"
            f"Please check that your metadata folder or file exists."
        )

    papers = []
    for i, rec in enumerate(records):
        try:
            raw_pdf_name = rec.get("raw_pdf_local_path")
            if raw_pdf_name:
                pdf_filename = Path(raw_pdf_name).name
            else:
                pdf_filename = f"{rec[FIELD_MAPPING['pdf_filename']]}.pdf"

            papers.append(
                PaperMetadata(
                    paper_id=str(rec[FIELD_MAPPING["paper_id"]]),
                    title=rec[FIELD_MAPPING["title"]],
                    authors=[a["name"] if isinstance(a, dict) else a for a in rec.get(FIELD_MAPPING["authors"], [])],
                    topic=rec.get(FIELD_MAPPING["topic"], "N/A"),
                    abstract=rec.get(FIELD_MAPPING["abstract"], ""),
                    pdf_filename=pdf_filename,
                    citation_count=rec.get("citation_count"),
                    publication_date=rec.get("publication_date"),
                )
            )
        except KeyError as e:
            raise KeyError(
                f"Record #{i} is missing field {e}. "
                f"Check FIELD_MAPPING against available JSON keys: {list(rec.keys())}"
            )

    return papers


def verify_pdfs_exist(papers: List[PaperMetadata], pdf_dir: str = "papers") -> List[str]:
    """Returns a list of paper_ids whose PDF files are missing on disk."""
    missing = []
    pdf_path = Path(pdf_dir)
    if not pdf_path.is_absolute():
        pdf_path = PROJECT_ROOT / pdf_dir

    for p in papers:
        if not (pdf_path / p.pdf_filename).exists():
            missing.append(p.paper_id)

    return missing