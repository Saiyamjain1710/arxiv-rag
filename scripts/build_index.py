import sys, json, uuid, os
from pathlib import Path
sys.path.append(".")

from qdrant_client.models import PointStruct
from app.retrieval.embedder import embed_batch, sparse_weights_to_qdrant_format
from app.retrieval.qdrant_store import create_collection_if_not_exists, upsert_chunks

CHUNKS_PATH = "data/processed_chunks.jsonl"
INDEXED_TRACKER_PATH = "data/indexed_chunk_ids.txt"
BATCH_SIZE = 12

NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")  # fixed, never change this


def deterministic_id(chunk_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE, chunk_id))


def get_already_indexed_ids() -> set[str]:
    if not os.path.exists(INDEXED_TRACKER_PATH):
        return set()
    with open(INDEXED_TRACKER_PATH, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def mark_as_indexed(chunk_ids: list[str]):
    # Ensure directory exists before writing
    Path(INDEXED_TRACKER_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(INDEXED_TRACKER_PATH, "a", encoding="utf-8") as f:
        for cid in chunk_ids:
            f.write(cid + "\n")


def load_chunks():
    records = []
    # Added encoding="utf-8" to handle special characters on Windows
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    create_collection_if_not_exists()
    records = load_chunks()

    already_indexed = get_already_indexed_ids()
    records_to_index = [r for r in records if r["chunk_id"] not in already_indexed]
    print(f"{len(already_indexed)} chunks already indexed, {len(records_to_index)} new chunks to embed and index.")

    for i in range(0, len(records_to_index), BATCH_SIZE):
        batch = records_to_index[i:i + BATCH_SIZE]
        texts = [r["text"] for r in batch]

        embeddings = embed_batch(texts)

        points = []
        for rec, dense_vec, sparse_w in zip(batch, embeddings["dense_vecs"], embeddings["sparse_weights"]):
            indices, values = sparse_weights_to_qdrant_format(sparse_w)
            points.append(PointStruct(
                id=deterministic_id(rec["chunk_id"]),
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
        mark_as_indexed([r["chunk_id"] for r in batch])
        print(f"   Indexed {min(i + BATCH_SIZE, len(records_to_index))}/{len(records_to_index)}")

    print("Done indexing.")


if __name__ == "__main__":
    main()