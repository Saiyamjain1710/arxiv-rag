import sys, json, os, gc
from pathlib import Path
sys.path.append(".")

from app.ingestion.metadata_loader import load_metadata, verify_pdfs_exist
from app.ingestion.pdf_parser import parse_pdf_to_markdown
from app.ingestion.chunker import chunk_markdown

METADATA_PATH = "data/metadata"
PDF_DIR = "data/raw_pdfs"
OUTPUT_PATH = "data/processed_chunks.jsonl"


def get_already_processed_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    ids = set()
    # Fixed: added encoding="utf-8" to handle special characters on Windows
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                try:
                    ids.add(json.loads(line)["paper_id"])
                except Exception:
                    continue
    return ids


def main():
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    try:
        papers = load_metadata(meta_dir=METADATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: Could not find metadata at '{METADATA_PATH}'.")
        print("Please check where your metadata JSON files are located and update METADATA_PATH.")
        return

    print(f"Loaded metadata for {len(papers)} papers.")

    missing = verify_pdfs_exist(papers, PDF_DIR)
    if missing:
        print(f"WARNING: {len(missing)} PDFs missing on disk: {missing[:5]}...")

    already_processed = get_already_processed_ids(OUTPUT_PATH)
    papers_to_process = [p for p in papers if p.paper_id not in already_processed]
    print(f"{len(already_processed)} papers already processed, {len(papers_to_process)} new papers to process.")

    total_chunks = 0
    with open(OUTPUT_PATH, "a", encoding="utf-8") as out_f:
        for i, paper in enumerate(papers_to_process):
            pdf_path = f"{PDF_DIR}/{paper.pdf_filename}"
            
            try:
                # OOM Guardrail: Skip abnormally large files (> 20 MB)
                file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
                if file_size_mb > 20.0:
                    print(f"  [{i+1}/{len(papers_to_process)}] SKIPPED {paper.paper_id}: File too large ({file_size_mb:.1f} MB)")
                    continue

                markdown = parse_pdf_to_markdown(pdf_path)
                chunks = chunk_markdown(markdown)
                
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

                print(f"  [{i+1}/{len(papers_to_process)}] {paper.paper_id}: {len(chunks)} chunks")

            except Exception as e:
                print(f"  [{i+1}/{len(papers_to_process)}] FAILED on {paper.paper_id}: {e}")
            finally:
                gc.collect()

    print(f"\nDone. {total_chunks} new chunks appended to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()