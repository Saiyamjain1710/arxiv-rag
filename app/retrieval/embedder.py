import torch
# --- Compatibility shim for transformers v5+ and FlagEmbedding ---
import transformers.utils.import_utils

if not hasattr(transformers.utils.import_utils, "is_torch_fx_available"):
    transformers.utils.import_utils.is_torch_fx_available = lambda: True
# -----------------------------------------------------------------

from FlagEmbedding import BGEM3FlagModel

# Global model cache variable (starts uninitialized)
_model_instance = None


def get_model() -> BGEM3FlagModel:
    """Lazy initializer: Loads BAAI/bge-m3 only when first requested."""
    global _model_instance
    if _model_instance is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[embedder] Loading BGEM3FlagModel on device: {device}...")
        _model_instance = BGEM3FlagModel(
            "BAAI/bge-m3", 
            use_fp16=(device == "cuda"), 
            device=device
        )
        print("[embedder] Model loaded successfully.")
    return _model_instance


def embed_batch(texts: list[str]) -> dict:
    """Returns dense vectors and sparse (lexical) weights for a batch of texts.
    dense_vecs: list of 1024-dim float vectors
    sparse_weights: list of {token_id: weight} dicts
    """
    model = get_model()
    output = model.encode(
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