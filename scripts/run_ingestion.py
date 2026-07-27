import sys, json
from pathlib import Path
sys.path.append(".")

from app.ingestion.metadata_loader import load_metadata, verify_pdfs_exist
from app.ingestion.pdf_parser import parse_pdf_to_markdown
from app.ingestion.chunker import chunk_markdown

# 1. Standardize paths under the data/ folder
METADATA_PATH = "data/metadata" # Or "data/metadata.json" if it's a single JSON file
PDF_DIR = "data/raw_pdfs"
OUTPUT_PATH = "data/processed_chunks.jsonl"


def main():
    # Ensure the output directory exists
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    # 2. Pass METADATA_PATH to load_metadata
    try:
        papers = load_metadata(meta_dir=METADATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: Could not find metadata at '{METADATA_PATH}'.")
        print("Please check where your metadata JSON files/file are located and update METADATA_PATH.")
        return

    print(f"Loaded metadata for {len(papers)} papers.")

    missing = verify_pdfs_exist(papers, PDF_DIR)
    if missing:
        print(f"WARNING: {len(missing)} PDFs missing on disk: {missing[:5]}...")

    total_chunks = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for i, paper in enumerate(papers):
            pdf_path = f"{PDF_DIR}/{paper.pdf_filename}"
            try:
                markdown = parse_pdf_to_markdown(pdf_path)
                chunks = chunk_markdown(markdown)
            except Exception as e:
                print(f"  [{i+1}/{len(papers)}] FAILED on {paper.paper_id}: {e}")
                continue

            for chunk_idx, chunk in enumerate(chunks):
                record = {
                    "chunk_id": f"{paper.paper_id}_{chunk_idx}",
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "authors": paper.authors,
                    "topic": paper.topic,
                    "section": chunk["metadata"],
                    "text": chunk["text"],
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

            print(f"  [{i+1}/{len(papers)}] {paper.paper_id}: {len(chunks)} chunks")

    print(f"\nDone. {total_chunks} total chunks written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()