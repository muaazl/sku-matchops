"""Deterministic BM25 lexical term-frequency tokenization and weighted token dictionary merging."""

from __future__ import annotations

import re
import zlib

BM25_TOKEN_PATTERN = re.compile(r"[\w\u0D80-\u0DFF\u0B80-\u0BFF]+", re.UNICODE)


def compute_bm25_tf_weights(
    text: str,
    k1: float = 1.2,
    b: float = 0.75,
    avg_len: float = 8.0,
) -> dict[int, float]:
    """
    Computes deterministic BM25 term-frequency saturation weights mapped to 31-bit CRC32 token IDs.
    Paired with Qdrant's server-side Modifier.IDF for hybrid BM25 retrieval.
    """
    if not text or not isinstance(text, str):
        return {}
    tokens = [t for t in BM25_TOKEN_PATTERN.findall(text.lower()) if len(t) > 1 or t.isdigit()]
    if not tokens:
        return {}

    doc_len = len(tokens)
    counts: dict[str, int] = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1

    norm_denom = k1 * (1.0 - b + b * (doc_len / max(avg_len, 1.0)))
    sparse: dict[int, float] = {}
    for tok, freq in counts.items():
        tok_id = (zlib.crc32(tok.encode("utf-8")) & 0x7FFFFFFF) or 1
        tf_val = float((freq * (k1 + 1.0)) / (freq + norm_denom))
        sparse[tok_id] = max(sparse.get(tok_id, 0.0), tf_val)
    return sparse


def merge_weighted_token_dicts(
    flat_dicts: list[dict[int, float]],
    name_idx: int,
    desc_idx: int | None,
    cat_idx: int | None,
    weights: tuple[float, float, float],
) -> dict[int, float]:
    """Combines weighted field token-score maps using max-pooling across weighted fields."""
    w_name, w_desc, w_cat = weights
    merged: dict[int, float] = {}
    for idx, weight in ((name_idx, w_name), (desc_idx, w_desc), (cat_idx, w_cat)):
        if idx is None:
            continue
        for tok_id, score in flat_dicts[idx].items():
            merged[tok_id] = max(merged.get(tok_id, 0.0), weight * score)
    return merged
