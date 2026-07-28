from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, PointStruct,
    SparseVector, Prefetch, FusionQuery, Fusion,
)
from app.config import settings

COLLECTION_NAME = "papers"

client = QdrantClient(url=settings.qdrant_url)


def create_collection_if_not_exists():
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"Collection '{COLLECTION_NAME}' already exists — skipping creation.")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )
    print(f"Created collection '{COLLECTION_NAME}'.")


def upsert_chunks(points: list[PointStruct]):
    client.upsert(collection_name=COLLECTION_NAME, points=points)


def hybrid_search(dense_vec, sparse_indices, sparse_values, limit: int = 15, topic_filter: str | None = None):
    """RRF fusion of dense + sparse search, done server-side inside Qdrant.
    Returns top `limit` fused results — this is the exact 'top-15' your
    architecture passes into the reranker next (Phase 4)."""
    query_filter = None
    if topic_filter:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        query_filter = Filter(must=[FieldCondition(key="topic", match=MatchValue(value=topic_filter))])

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=dense_vec, using="dense", limit=50, filter=query_filter),
            Prefetch(query=SparseVector(indices=sparse_indices, values=sparse_values), using="sparse", limit=50, filter=query_filter),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=limit,
    )
    return results.points