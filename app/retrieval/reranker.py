import torch
from FlagEmbedding import FlagReranker

_device = "cuda" if torch.cuda.is_available() else "cpu"

_reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=(_device == "cuda"))

# --- Clean Patch for FlagEmbedding + Transformers (XLMRobertaTokenizer) ---
if hasattr(_reranker, "tokenizer") and not hasattr(
    _reranker.tokenizer, "prepare_for_model"
):

    def _prepare_for_model_patch(
        self,
        ids,
        pair_ids=None,
        add_special_tokens=True,
        padding=False,
        truncation=None,
        max_length=None,
        stride=0,
        pad_to_multiple_of=None,
        return_tensors=None,
        return_token_type_ids=None,
        return_attention_mask=None,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        return_offsets_mapping=False,
        verbose=True,
        **kwargs,
    ):
        # Handle pre-tokenized sequence list (IDs) passed by FlagEmbedding
        if isinstance(ids, list):
            # If pair_ids were provided, concatenate sequence pairs
            if pair_ids is not None:
                encoded_inputs = {"input_ids": ids + pair_ids}
            else:
                encoded_inputs = {"input_ids": ids}

            return self.pad(
                encoded_inputs,
                padding=padding,
                max_length=max_length,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors=return_tensors,
                verbose=verbose,
            )

        # Fallback for raw text strings
        return self._encode_plus(
            ids,
            second_ids=pair_ids,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding,
            truncation_strategy=truncation,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            verbose=verbose,
            **kwargs,
        )

    # Attach patched method to the tokenizer instance
    _reranker.tokenizer.prepare_for_model = _prepare_for_model_patch.__get__(
        _reranker.tokenizer, type(_reranker.tokenizer)
    )
# ------------------------------------------------------------


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """candidates: list of dicts with a 'text' key (from Qdrant hybrid_search payloads).

    Returns the same dicts, re-sorted, with a 'rerank_score' added, trimmed to
    top_k.
    """
    if not candidates:
        return []

    pairs = [[query, c["text"]] for c in candidates]
    scores = _reranker.compute_score(pairs, normalize=True)

    if isinstance(scores, float):
        scores = [scores]

    for c, s in zip(candidates, scores):
        c["rerank_score"] = s

    ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]