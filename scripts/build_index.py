import sys, json, uuid
sys.path.append(".")

from qdrant_client.models import PointStruct
from app.retrieval.embedder import embed_batch, sparse_weights_to_qdrant_format
from app.retrieval.qdrant_store import create_collection_if_not_exists, upsert_chunks

CHUNKS_PATH = "data/processed_chunks.jsonl"
BATCH_SIZE = 12


def load_chunks():
    records = []
    # Added encoding="utf-8" to handle special characters on Windows
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def main():
    create_collection_if_not_exists()
    records = load_chunks()
    print(f"Embedding and indexing {len(records)} chunks...")

    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        texts = [r["text"] for r in batch]

        embeddings = embed_batch(texts)

        points = []
        for rec, dense_vec, sparse_w in zip(batch, embeddings["dense_vecs"], embeddings["sparse_weights"]):
            indices, values = sparse_weights_to_qdrant_format(sparse_w)
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector={"dense": dense_vec.tolist(), "sparse": {"indices": indices, "values": values}},
                payload={
                    "chunk_id": rec["chunk_id"],
                    "paper_id": rec["paper_id"],
                    "title": rec["title"],
                    "authors": rec["authors"],
                    "topic": rec["topic"],
                    "section": rec.get("section", {}),
                    "text": rec["text"],
                },
            ))

        upsert_chunks(points)
        print(f"   Indexed {min(i + BATCH_SIZE, len(records))}/{len(records)}")

    print("Done indexing.")


if __name__ == "__main__":
    main()