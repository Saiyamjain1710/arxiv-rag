import uuid
import arxiv
from pathlib import Path
from qdrant_client.models import PointStruct

from app.ingestion.pdf_parser import parse_pdf_to_markdown
from app.ingestion.chunker import chunk_markdown
from app.retrieval.embedder import embed_batch, sparse_weights_to_qdrant_format
from app.retrieval.qdrant_store import upsert_chunks

FETCHED_PDF_DIR = "data/fetched_pdfs"
Path(FETCHED_PDF_DIR).mkdir(parents=True, exist_ok=True)


def search_arxiv(query: str, max_results: int = 3):
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    return list(client.results(search))


def _download_pdf(result) -> tuple[str, str]:
    paper_id = result.get_short_id()
    filename = f"{paper_id.replace('/', '_')}.pdf"
    result.download_pdf(dirpath=FETCHED_PDF_DIR, filename=filename)
    return paper_id, f"{FETCHED_PDF_DIR}/{filename}"


def fetch_and_index(query: str, max_results: int = 3) -> list[str]:
    """Searches arXiv live, downloads matching papers, parses + chunks + embeds
    + indexes them into the same Qdrant collection, and returns their paper_ids
    so the retrieval step can mark them as trusted for THIS session only."""
    results = search_arxiv(query, max_results)
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
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector={"dense": dense_vec.tolist(), "sparse": {"indices": indices, "values": values}},
                    payload={
                        "chunk_id": f"{paper_id}_{idx}",
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
            print(f"  Fetched + indexed: {paper_id} — {r.title[:60]}")

        except Exception as e:
            print(f"  Failed to fetch/index '{r.title[:60]}': {e}")

    return new_ids