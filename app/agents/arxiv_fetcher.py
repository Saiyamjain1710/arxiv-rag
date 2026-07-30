import uuid
import re
import time
import arxiv
from pathlib import Path
from qdrant_client.models import PointStruct

from app.ingestion.pdf_parser import parse_pdf_to_markdown
from app.ingestion.chunker import chunk_markdown
from app.retrieval.embedder import embed_batch, sparse_weights_to_qdrant_format
from app.retrieval.qdrant_store import upsert_chunks
from app.retrieval.ids import deterministic_id

FETCHED_PDF_DIR = "data/fetched_pdfs"
Path(FETCHED_PDF_DIR).mkdir(parents=True, exist_ok=True)

NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")  # fixed, never change this


def deterministic_id(chunk_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE, chunk_id))


# Configure global client with required rate limits and exponential backoff retries
arxiv_client = arxiv.Client(
    page_size=25,
    delay_seconds=3.0,  # Adheres strictly to arXiv terms (min 3s delay)
    num_retries=3       # Auto-retries transient 429/503 errors
)


def sanitize_arxiv_query(query: str) -> str:
    """Cleans punctuation, question marks, and limits search terms for arXiv API."""
    cleaned = re.sub(r'[^\w\s]', '', query).strip()
    words = cleaned.split()
    # Keep up to 6 key terms to avoid API errors on complex conversational strings
    return " ".join(words[:6])


def search_arxiv(query: str, max_results: int = 3):
    clean_query = sanitize_arxiv_query(query)
    if not clean_query:
        return []

    search = arxiv.Search(
        query=clean_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    try:
        return list(arxiv_client.results(search))
    except arxiv.HTTPError as e:
        print(f"[arxiv_fetcher] arXiv API HTTP Error ({e.status}): {e}")
        return []
    except Exception as e:
        print(f"[arxiv_fetcher] Failed to query arXiv: {e}")
        return []


def _download_pdf(result) -> tuple[str, str]:
    paper_id = result.get_short_id()
    filename = f"{paper_id.replace('/', '_')}.pdf"
    pdf_path = f"{FETCHED_PDF_DIR}/{filename}"

    # Avoid re-downloading if the file was already fetched in a previous request
    if Path(pdf_path).exists():
        return paper_id, pdf_path

    result.download_pdf(dirpath=FETCHED_PDF_DIR, filename=filename)
    return paper_id, pdf_path


def fetch_and_index(query: str, max_results: int = 3) -> list[str]:
    """Searches arXiv live, downloads matching papers, parses + chunks + embeds
    + indexes them into the same Qdrant collection, and returns their paper_ids
    so the retrieval step can mark them as trusted for THIS session only."""
    
    try:
        results = search_arxiv(query, max_results)
    except Exception as e:
        print(f"[arxiv_fetcher] Error during arXiv search execution: {e}")
        return []

    new_ids = []

    for r in results:
        try:
            paper_id, pdf_path = _download_pdf(r)
            markdown = parse_pdf_to_markdown(pdf_path)
            chunks = chunk_markdown(markdown)
            if not chunks:
                continue

            texts = [c["text"] for c in chunks]
            emb = embed_batch(texts)

            points = []
            for idx, (chunk, dense_vec, sparse_w) in enumerate(zip(chunks, emb["dense_vecs"], emb["sparse_weights"])):
                indices, values = sparse_weights_to_qdrant_format(sparse_w)
                chunk_id = f"{paper_id}_{idx}"
                points.append(PointStruct(
                    id=deterministic_id(chunk_id),
                    vector={"dense": dense_vec.tolist(), "sparse": {"indices": indices, "values": values}},
                    payload={
                        "chunk_id": chunk_id,
                        "paper_id": paper_id,
                        "title": r.title,
                        "authors": [a.name for a in r.authors],
                        "topic": "arxiv_fetched",
                        "section": chunk["metadata"],
                        "text": chunk["text"],
                    },
                ))

            upsert_chunks(points)
            new_ids.append(paper_id)
            print(f"   Fetched + indexed: {paper_id} — {r.title[:60]}")

            # Brief pause between downloads to prevent IP throttling
            time.sleep(1.0)

        except Exception as e:
            print(f"   Failed to fetch/index '{r.title[:60]}': {e}")

    return new_ids