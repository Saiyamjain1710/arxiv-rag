import torch
from FlagEmbedding import BGEM3FlagModel

_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[embedder] Using device: {_device}")

_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=(_device == "cuda"), device=_device)


def embed_batch(texts: list[str]) -> dict:
    """Returns dense vectors and sparse (lexical) weights for a batch of texts.
    dense_vecs: list of 1024-dim float vectors
    sparse_weights: list of {token_id: weight} dicts
    """
    output = _model.encode(
        texts,
        batch_size=12,
        max_length=1024,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    return {
        "dense_vecs": output["dense_vecs"],
        "sparse_weights": output["lexical_weights"],
    }


def sparse_weights_to_qdrant_format(weights: dict) -> tuple[list[int], list[float]]:
    """Converts {token_id: weight} into the (indices, values) format Qdrant expects."""
    indices = [int(k) for k in weights.keys()]
    values = [float(v) for v in weights.values()]
    return indices, values