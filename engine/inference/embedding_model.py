"""ONNX Runtime bi-encoder embedding model with hybrid Dense + BM25 + Sparse support."""

from __future__ import annotations

import logging
import os
import sys
import threading
from collections import OrderedDict
from pathlib import Path

import joblib
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from engine.core import config
from engine.core.observability import ONNX_INFERENCE_SECONDS
from engine.core.onnx_factory import create_onnx_session, filter_onnx_inputs
from engine.preprocessing.bm25_tokenizer import (
    compute_bm25_tf_weights,
    merge_weighted_token_dicts,
)

logger = logging.getLogger("matchops.inference.embedding")


class EmbeddingModel:
    """ONNX bi-encoder generating normalized dense, BM25 TF, and optional sparse vectors."""

    def __init__(self) -> None:
        self._cache_lock = threading.RLock()
        self._str_cache: OrderedDict[
            str, tuple[np.ndarray, dict[int, float], dict[int, float] | None]
        ] = OrderedDict()
        self._load_embedding_model()

    def _load_embedding_model(self) -> None:
        """Loads the ONNX bi-encoder session and tokenizer with fail-fast validation."""
        emb_onnx = Path(config.EMBEDDING_ONNX)
        if not emb_onnx.exists():
            raise FileNotFoundError(
                f"[FAIL-FAST] Required ONNX embedding model missing at '{emb_onnx}'. "
                "Run 'python -m scripts.sync --models sync --model-target embedding' first."
            )

        prec_label = getattr(config, "EMBEDDING_ACTIVE_PRECISION", "int8").upper()
        logger.info(
            "[EMBED] Loading Embedding (%s) %s ONNX from %s...",
            config.EMBEDDING_MODEL,
            prec_label,
            emb_onnx,
        )
        self.emb_manifest = config.read_json_manifest(config.EMBEDDING_MANIFEST)
        manifest_pooling = str(self.emb_manifest.get("pooling", "last_token")).lower()
        env_pooling = str(getattr(config, "EMBEDDING_POOLING", "auto")).lower()
        self._emb_pooling = (
            env_pooling if env_pooling in ("cls", "mean", "last_token") else manifest_pooling
        )
        self.embedding_dim = int(self.emb_manifest.get("embedding_dim", config.EMBEDDING_DIM))

        try:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(emb_onnx.parent),
                    local_files_only=True,
                    trust_remote_code=True,
                    fix_mistral_regex=True,
                )
            except TypeError:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(emb_onnx.parent),
                    local_files_only=True,
                    trust_remote_code=True,
                    fix_mistral_regex=False,
                )
            if getattr(self.tokenizer, "padding_side", None) == "left":
                self.tokenizer.padding_side = "right"
            self.bi_session, _ = create_onnx_session(
                emb_onnx, getattr(config, "EMBEDDING_ORT_PROVIDERS", None)
            )
        except Exception as exc:
            raise RuntimeError(
                f"[FAIL-FAST] Failed to load ONNX embedding model '{emb_onnx}': {exc}"
            ) from exc

        session_outputs = (
            [o.name for o in self.bi_session.get_outputs()]
            if hasattr(self.bi_session, "get_outputs")
            else []
        )
        self.has_learned_sparse = bool(
            self.emb_manifest.get("has_learned_sparse", False)
            or "sparse_weights" in session_outputs
            or len(session_outputs) > 1
        )
        logger.info(
            "[EMBED] Embedding %s ONNX loaded (pooling=%s, bm25=true, learned_sparse=%s).",
            prec_label,
            self._emb_pooling,
            self.has_learned_sparse,
        )

    @staticmethod
    def pool_hidden_states(
        hidden: np.ndarray, attention_mask: np.ndarray, pooling: str = "last_token"
    ) -> np.ndarray:
        """Pools 3D sequence hidden states `(B, L, D)` into 2D embeddings `(B, D)`."""
        if pooling == "cls":
            return hidden[:, 0, :]
        if pooling == "mean":
            mask = np.expand_dims(attention_mask, -1).astype(hidden.dtype)
            denom = np.clip(mask.sum(axis=1), 1e-9, None)
            return (hidden * mask).sum(axis=1) / denom
        has_tok = attention_mask.sum(axis=1) > 0
        rev_argmax = np.argmax(attention_mask[:, ::-1], axis=1)
        seq_lens = np.where(has_tok, attention_mask.shape[1] - 1 - rev_argmax, 0)
        return hidden[np.arange(hidden.shape[0]), seq_lens]

    def warmup(self) -> None:
        """Runs a tiny dummy forward pass to pre-allocate ONNX Runtime buffers."""
        if getattr(self, "bi_session", None):
            dummy = self.tokenizer("warmup", return_tensors="np")
            inputs = filter_onnx_inputs(
                self.bi_session, {k: v.astype(np.int64) for k, v in dummy.items()}
            )
            self.bi_session.run(None, inputs)
            logger.info("[EMBED] Embedding (ONNX) warmed up.")

    def encode(
        self, texts: list[str], batch_size: int = config.ONNX_BATCH_SIZE
    ) -> dict[str, np.ndarray | list[dict]]:
        """Encodes texts into normalized `dense` vectors, `bm25` weights, and optional `sparse` weights."""
        if not texts:
            return {"dense": np.array([]), "bm25": [], "sparse": []}

        if not getattr(self, "bi_session", None):
            raise RuntimeError("[FAIL-FAST] Embedding ONNX session is not initialized.")

        cache_lock = getattr(self, "_cache_lock", None)
        if cache_lock is None:
            cache_lock = threading.RLock()
            self._cache_lock = cache_lock
        if not hasattr(self, "_str_cache"):
            self._str_cache = OrderedDict()

        resolved: dict[str, tuple[np.ndarray, dict[int, float], dict[int, float] | None]] = {}
        missing_texts: list[str] = []
        missing_set: set[str] = set()
        with cache_lock:
            for text in texts:
                if not text or not str(text).strip():
                    continue
                if text in self._str_cache:
                    self._str_cache.move_to_end(text)
                    resolved[text] = self._str_cache[text]
                elif text not in missing_set:
                    missing_texts.append(text)
                    missing_set.add(text)

        pooling = getattr(self, "_emb_pooling", getattr(self, "pooling", "last_token"))
        unused_tokens = (
            set(getattr(self.tokenizer, "all_special_ids", ()))
            if getattr(self, "tokenizer", None)
            else {0, 1, 2, 3}
        )

        if missing_texts:
            max_cache_size = int(os.getenv("STR_CACHE_MAX_SIZE", "4096"))
            raw_ids_list = None
            if pooling == "last_token":
                try:
                    raw_tok = self.tokenizer(
                        missing_texts, padding=False, truncation=True, max_length=512
                    )
                    cand_ids = (
                        raw_tok["input_ids"]
                        if isinstance(raw_tok, dict) or hasattr(raw_tok, "__getitem__")
                        else None
                    )
                    if (
                        cand_ids is not None
                        and len(cand_ids) == len(missing_texts)
                        and all(hasattr(x, "__len__") and len(x) > 0 for x in cand_ids)
                    ):
                        raw_ids_list = cand_ids
                except Exception:
                    raw_ids_list = None

            if raw_ids_list is not None:
                len_groups: dict[int, list[int]] = {}
                long_indices: list[int] = []
                for idx_t, tok_ids in enumerate(raw_ids_list):
                    t_len = len(tok_ids)
                    if t_len <= 24:
                        len_groups.setdefault(t_len, []).append(idx_t)
                    else:
                        long_indices.append(idx_t)

                sub_batches: list[list[int]] = []
                for seq_len in sorted(len_groups.keys()):
                    g_indices = len_groups[seq_len]
                    for start_g in range(0, len(g_indices), batch_size):
                        sub_batches.append(g_indices[start_g : start_g + batch_size])

                if long_indices:
                    long_indices.sort(key=lambda idx_t: len(raw_ids_list[idx_t]))
                    for start_l in range(0, len(long_indices), batch_size):
                        sub_batches.append(long_indices[start_l : start_l + batch_size])

                batch_iter = sub_batches
                if len(missing_texts) > batch_size:
                    batch_iter = tqdm(
                        sub_batches,
                        desc=f"[EMBED] Encoding ({len(missing_texts)} texts)",
                        unit="batch",
                        mininterval=1.0,
                        leave=False,
                    )

                pad_id = int(getattr(self.tokenizer, "pad_token_id", 0) or 0)
                for sub_idx_list in batch_iter:
                    max_sub_len = max(len(raw_ids_list[idx_t]) for idx_t in sub_idx_list)
                    min_sub_len = min(len(raw_ids_list[idx_t]) for idx_t in sub_idx_list)
                    if max_sub_len == min_sub_len:
                        ids_np = np.array(
                            [raw_ids_list[idx_t] for idx_t in sub_idx_list], dtype=np.int64
                        )
                        mask_np = np.ones_like(ids_np, dtype=np.int64)
                    else:
                        ids_np = np.full((len(sub_idx_list), max_sub_len), pad_id, dtype=np.int64)
                        mask_np = np.zeros((len(sub_idx_list), max_sub_len), dtype=np.int64)
                        for b_i, idx_t in enumerate(sub_idx_list):
                            row_ids = raw_ids_list[idx_t]
                            r_len = len(row_ids)
                            ids_np[b_i, :r_len] = row_ids
                            mask_np[b_i, :r_len] = 1
                    ort_inputs = filter_onnx_inputs(
                        self.bi_session, {"input_ids": ids_np, "attention_mask": mask_np}
                    )
                    with ONNX_INFERENCE_SECONDS.labels(model="embed").time():
                        outputs = self.bi_session.run(None, ort_inputs)
                    dense_vecs = outputs[0]
                    if dense_vecs.ndim == 3:
                        dense_vecs = self.pool_hidden_states(dense_vecs, mask_np, pooling=pooling)
                    norms = np.linalg.norm(dense_vecs, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1e-12, norms)
                    normed_dense = dense_vecs / norms

                    sub_sparse: list[dict[int, float] | None] = [None] * len(sub_idx_list)
                    if len(outputs) > 1:
                        sparse_weights = outputs[1]
                        for b in range(len(sub_idx_list)):
                            d: dict[int, float] = {}
                            for tok_id, w in zip(ids_np[b], sparse_weights[b]):
                                tok_id = int(tok_id)
                                if tok_id not in unused_tokens and w > 0:
                                    d[tok_id] = max(d.get(tok_id, 0.0), float(w))
                            sub_sparse[b] = d

                    with cache_lock:
                        for local_i, orig_t_idx in enumerate(sub_idx_list):
                            text_item = missing_texts[orig_t_idx]
                            entry = (
                                normed_dense[local_i],
                                compute_bm25_tf_weights(text_item),
                                sub_sparse[local_i],
                            )
                            resolved[text_item] = entry
                            self._str_cache[text_item] = entry
                            self._str_cache.move_to_end(text_item)
                        while len(self._str_cache) > max_cache_size:
                            self._str_cache.popitem(last=False)
            else:
                all_dense = []
                all_bm25: list[dict[int, float]] = []
                all_learned_sparse: list[dict[int, float] | None] = []

                batch_starts = range(0, len(missing_texts), batch_size)
                if len(missing_texts) > batch_size:
                    batch_starts = tqdm(
                        batch_starts,
                        desc=f"[EMBED] Encoding ({len(missing_texts)} texts)",
                        unit="batch",
                        mininterval=1.0,
                        leave=False,
                    )

                for i in batch_starts:
                    batch = missing_texts[i : i + batch_size]
                    inputs = self.tokenizer(
                        batch, padding=True, truncation=True, max_length=512, return_tensors="np"
                    )
                    ort_inputs = filter_onnx_inputs(
                        self.bi_session, {k: v.astype(np.int64) for k, v in inputs.items()}
                    )

                    with ONNX_INFERENCE_SECONDS.labels(model="embed").time():
                        outputs = self.bi_session.run(None, ort_inputs)
                    dense_vecs = outputs[0]

                    if dense_vecs.ndim == 3:
                        dense_vecs = self.pool_hidden_states(
                            dense_vecs, inputs["attention_mask"], pooling=pooling
                        )

                    norms = np.linalg.norm(dense_vecs, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1e-12, norms)
                    all_dense.append(dense_vecs / norms)

                    for text_item in batch:
                        all_bm25.append(compute_bm25_tf_weights(text_item))

                    if len(outputs) > 1:
                        sparse_weights = outputs[1]
                        for b in range(len(batch)):
                            d = {}
                            for tok_id, w in zip(inputs["input_ids"][b], sparse_weights[b]):
                                tok_id = int(tok_id)
                                if tok_id not in unused_tokens and w > 0:
                                    d[tok_id] = max(d.get(tok_id, 0.0), float(w))
                            all_learned_sparse.append(d)
                    else:
                        all_learned_sparse.extend([None] * len(batch))

                flat_dense = np.vstack(all_dense) if all_dense else np.array([])
                with cache_lock:
                    for i, text in enumerate(missing_texts):
                        entry = (flat_dense[i], all_bm25[i], all_learned_sparse[i])
                        resolved[text] = entry
                        self._str_cache[text] = entry
                        self._str_cache.move_to_end(text)
                    while len(self._str_cache) > max_cache_size:
                        self._str_cache.popitem(last=False)

        first_non_empty = next(iter(resolved.values()), None)
        dim = first_non_empty[0].shape[0] if first_non_empty is not None else config.EMBEDDING_DIM
        zero_dense = np.zeros(dim, dtype=np.float32)
        has_sparse = bool(getattr(self, "has_learned_sparse", False)) or any(
            v[2] is not None for v in resolved.values()
        )
        out_dense = np.empty((len(texts), dim), dtype=np.float32)
        out_bm25: list[dict[int, float]] = []
        out_sparse: list[dict[int, float]] = []

        for i, text in enumerate(texts):
            if not text or not str(text).strip() or text not in resolved:
                out_dense[i] = zero_dense
                out_bm25.append({})
                if has_sparse:
                    out_sparse.append({})
                continue
            d, bm25_d, learned_d = resolved[text]
            out_dense[i] = d
            out_bm25.append(bm25_d)
            if has_sparse:
                out_sparse.append(learned_d if learned_d is not None else {})

        return {
            "dense": out_dense,
            "bm25": out_bm25,
            "sparse": out_sparse,
        }

    def encode_query(self, text: str) -> dict[str, np.ndarray | list[dict]]:
        """Encodes a single query string."""
        return self.encode([text], batch_size=1)

    def embed_queries(
        self, queries: list[str], domain: str | None = None, cache_key: str = "sku_queries"
    ) -> dict[str, np.ndarray | list[dict]]:
        """Batch-encodes SKU queries with logging for classifier workflows."""
        if not queries:
            return {"dense": np.array([]), "bm25": [], "sparse": []}
        logger.info("[EMBED] Encoding %d SKU queries...", len(queries))
        result = self.encode(queries)
        logger.info("[EMBED] Done encoding %d queries.", len(queries))
        return result

    def embed_weighted_sku(
        self,
        names: list[str],
        descriptions: list[str],
        categories: list[str],
        weights: tuple = config.CLASSIFIER_WEIGHTS,
    ) -> dict[str, np.ndarray | list[dict]]:
        """Embeds SKUs using field-weighted averaging across Name, Description, and Category."""
        if not names:
            return {"dense": np.array([]), "bm25": [], "sparse": []}

        n = len(names)
        w_name, w_desc, w_cat = weights

        def _clean(lst):
            return [
                str(v).strip()
                if v and str(v).strip().lower() not in ("nan", "none", "<na>")
                else ""
                for v in lst
            ]

        def _align_col(lst) -> list[str]:
            vals = _clean(lst) if lst else []
            if len(vals) < n:
                vals.extend([""] * (n - len(vals)))
            return vals[:n]

        names_clean = _clean(names)
        descs_clean = _align_col(descriptions)
        cats_clean = _align_col(categories)

        flat_texts = []
        name_offsets = []
        desc_offsets = []
        cat_offsets = []

        for i in range(n):
            name_offsets.append(len(flat_texts))
            flat_texts.append(names_clean[i])

        for lst, offsets in ((descs_clean, desc_offsets), (cats_clean, cat_offsets)):
            if any(lst):
                for val in lst:
                    if val:
                        offsets.append(len(flat_texts))
                        flat_texts.append(val)
                    else:
                        offsets.append(None)
            else:
                offsets.extend([None] * n)

        encoded = self.encode(flat_texts)
        flat_dense = encoded["dense"]
        flat_bm25 = encoded.get("bm25") or [compute_bm25_tf_weights(t) for t in flat_texts]
        flat_sparse = encoded.get("sparse") or []
        has_sparse = bool(flat_sparse)

        all_dense = []
        all_bm25 = []
        all_sparse = []

        for i in range(n):
            name_idx = name_offsets[i]
            desc_idx = desc_offsets[i]
            cat_idx = cat_offsets[i]

            weighted_vec = w_name * flat_dense[name_idx]
            if desc_idx is not None:
                weighted_vec = weighted_vec + w_desc * flat_dense[desc_idx]
            if cat_idx is not None:
                weighted_vec = weighted_vec + w_cat * flat_dense[cat_idx]

            norm = np.linalg.norm(weighted_vec)
            if norm > 0:
                weighted_vec = weighted_vec / norm
            all_dense.append(weighted_vec)

            all_bm25.append(
                merge_weighted_token_dicts(flat_bm25, name_idx, desc_idx, cat_idx, weights)
            )
            if has_sparse:
                all_sparse.append(
                    merge_weighted_token_dicts(flat_sparse, name_idx, desc_idx, cat_idx, weights)
                )

        return {"dense": np.vstack(all_dense), "bm25": all_bm25, "sparse": all_sparse}

    def embed_dictionary_incremental(
        self, domain: str, dict_key: str, keywords: list[str]
    ) -> dict[str, np.ndarray | list[dict]]:
        """Encodes a dictionary of keywords incrementally with model-scoped disk caching."""
        cache_dir = Path(config.CACHE_DIR)
        cache_path = cache_dir / f"{domain}_{dict_key}_incremental.pkl"
        cache_dir.mkdir(parents=True, exist_ok=True)

        item_cache: dict = {}
        if cache_path.exists():
            try:
                loaded = joblib.load(cache_path)
                if (
                    isinstance(loaded, dict)
                    and loaded.get("embedding_model") == config.EMBEDDING_MODEL
                    and isinstance(loaded.get("items"), dict)
                ):
                    item_cache = loaded["items"]
            except Exception:
                item_cache = {}

        expect_sparse = getattr(self, "has_learned_sparse", False)
        missing_keywords = [
            kw
            for kw in keywords
            if kw not in item_cache
            or "bm25" not in item_cache[kw]
            or (expect_sparse and "sparse" not in item_cache[kw])
        ]

        if missing_keywords:
            logger.info(
                "[EMBED] %s/%s: %d new items to encode (%d cached).",
                domain,
                dict_key,
                len(missing_keywords),
                len(keywords) - len(missing_keywords),
            )
            all_dense, all_bm25, all_sparse = [], [], []
            batch_size = config.ONNX_BATCH_SIZE
            batches = [
                missing_keywords[i : i + batch_size]
                for i in range(0, len(missing_keywords), batch_size)
            ]
            num_batches = len(batches)
            for idx, batch in enumerate(batches, 1):
                new_embs = self.encode(batch)
                all_dense.extend(new_embs["dense"])
                all_bm25.extend(new_embs["bm25"])
                if new_embs.get("sparse"):
                    all_sparse.extend(new_embs["sparse"])
                if idx % 5 == 0 or idx == num_batches:
                    logger.info(
                        "[EMBED] %s/%s: Batch %d/%d encoded (%d/%d items)...",
                        domain,
                        dict_key,
                        idx,
                        num_batches,
                        min(idx * batch_size, len(missing_keywords)),
                        len(missing_keywords),
                    )
                    sys.stdout.flush()

            has_new_sparse = len(all_sparse) == len(missing_keywords)
            for i, kw in enumerate(missing_keywords):
                entry: dict[str, object] = {"dense": all_dense[i], "bm25": all_bm25[i]}
                if has_new_sparse:
                    entry["sparse"] = all_sparse[i]
                item_cache[kw] = entry

            joblib.dump(
                {"embedding_model": config.EMBEDDING_MODEL, "items": item_cache},
                cache_path,
            )
        else:
            logger.info(
                "[EMBED] %s/%s: all %d items served from cache.", domain, dict_key, len(keywords)
            )

        dense_list = [item_cache[kw]["dense"] for kw in keywords]
        bm25_list = [item_cache[kw]["bm25"] for kw in keywords]
        sparse_list = (
            [item_cache[kw]["sparse"] for kw in keywords]
            if keywords and all("sparse" in item_cache[kw] for kw in keywords)
            else []
        )

        return {"dense": np.array(dense_list), "bm25": bm25_list, "sparse": sparse_list}
