from concurrent.futures import ThreadPoolExecutor
import logging
import os
import sys
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
from transformers import AutoTokenizer

from engine import config

logger = logging.getLogger("matchops.embedder")

class EmbeddingEngine:
    """
    Hybrid embedding engine for BGE-M3 + Cross-Encoder.

    Dense vectors  → ONNX Runtime (fast, memory-mapped, CPU)
    Sparse vectors → ONNX Runtime (lexical_weights from custom export)
    Cross-encoder  → ONNX Runtime; falls back to sentence-transformers CrossEncoder
    """

    def __init__(self):
        self._load_models()

    def _load_models(self):
        """Loads all models in parallel using ONNX Runtime for dense/sparse/cross-encoder."""

        sess_options = ort.SessionOptions()
        sess_options.add_session_config_entry("session.use_mmap_for_weights", "1")
        sess_options.intra_op_num_threads = config.MAX_CPU_CORES
        sess_options.inter_op_num_threads = config.MAX_CPU_CORES

        def _load_bi_encoder():
            logger.info(f"[EMBED] Loading BGE-M3 Hybrid Bi-Encoder (ONNX) from {config.BI_ENCODER_ONNX}...")
            self.flag_model = None
            
            # Auto-export if missing
            if not os.path.exists(config.BI_ENCODER_ONNX):
                try:
                    from engine.export_onnx import export_bge_m3
                    logger.info("[EMBED] BGE-M3 ONNX not found. Initiating on-demand export...")
                    export_bge_m3()
                except Exception as exp_err:
                    logger.warning(f"[EMBED] Auto-export BGE-M3 failed ({exp_err}); will attempt PyTorch fallback.")

            try:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        os.path.dirname(config.BI_ENCODER_ONNX), local_files_only=True, fix_mistral_regex=True
                    )
                except Exception:
                    self.tokenizer = AutoTokenizer.from_pretrained(config.BI_ENCODER_MODEL, fix_mistral_regex=True)

                self.bi_session = ort.InferenceSession(
                    config.BI_ENCODER_ONNX, sess_options, providers=['CPUExecutionProvider']
                )
                logger.info("[EMBED] BGE-M3 ONNX hybrid bi-encoder loaded (dense + sparse).")
            except Exception as e:
                logger.warning(f"[EMBED] Bi-Encoder ONNX unavailable ({e}); falling back to PyTorch FlagEmbedding...")
                try:
                    from FlagEmbedding import BGEM3FlagModel
                    self.flag_model = BGEM3FlagModel(config.BI_ENCODER_MODEL, use_fp16=False)
                    logger.info("[EMBED] PyTorch BGEM3FlagModel loaded successfully as fallback.")
                except Exception as fe_err:
                    logger.error(f"[EMBED] PyTorch BGEM3FlagModel fallback failed: {fe_err}")
                self.bi_session = None

        def _load_cross_encoder():
            logger.info(f"[EMBED] Loading BGE-Reranker Cross-Encoder (ONNX) from {config.CROSS_ENCODER_ONNX}...")
            self.cross_encoder_fallback = None
            
            # Auto-export if missing
            if not os.path.exists(config.CROSS_ENCODER_ONNX):
                try:
                    from engine.export_onnx import export_bge_reranker
                    logger.info("[EMBED] BGE-Reranker ONNX not found. Initiating on-demand export...")
                    export_bge_reranker()
                except Exception as exp_err:
                    logger.warning(f"[EMBED] Auto-export BGE-Reranker failed ({exp_err}); will attempt PyTorch fallback.")

            try:
                try:
                    self.rerank_tokenizer = AutoTokenizer.from_pretrained(
                        os.path.dirname(config.CROSS_ENCODER_ONNX), local_files_only=True, fix_mistral_regex=True
                    )
                except Exception:
                    self.rerank_tokenizer = AutoTokenizer.from_pretrained(config.CROSS_ENCODER_MODEL, fix_mistral_regex=True)

                self.cross_session = ort.InferenceSession(
                    config.CROSS_ENCODER_ONNX, sess_options, providers=['CPUExecutionProvider']
                )
                logger.info("[EMBED] Cross-Encoder ONNX loaded.")
            except Exception as e:
                logger.warning(f"[EMBED] Cross-Encoder ONNX unavailable ({e}); falling back to sentence-transformers.")
                try:
                    from sentence_transformers import CrossEncoder
                    self.cross_encoder_fallback = CrossEncoder(config.CROSS_ENCODER_MODEL, device="cpu")
                    logger.info("[EMBED] sentence-transformers CrossEncoder loaded successfully.")
                except Exception as e2:
                    logger.error(f"[EMBED] sentence-transformers CrossEncoder also failed: {e2}")
                    self.cross_encoder_fallback = None
                self.cross_session = None

        with ThreadPoolExecutor(max_workers=2) as executor:
            fut_bi = executor.submit(_load_bi_encoder)
            fut_cross = executor.submit(_load_cross_encoder)
            fut_bi.result()
            fut_cross.result()

    def warmup(self):
        """Pre-allocates ONNX buffers."""
        if self.bi_session:
            dummy = self.tokenizer("warmup", return_tensors="np")
            inputs = {k: v.astype(np.int64) for k, v in dummy.items()}
            self.bi_session.run(None, inputs)
            logger.info("[EMBED] Bi-Encoder (ONNX dense+sparse) warmed up.")

        if self.cross_session:
            dummy = self.rerank_tokenizer("warmup", "text", return_tensors="np")
            inputs = {k: v.astype(np.int64) for k, v in dummy.items()}
            self.cross_session.run(None, inputs)
            logger.info("[EMBED] Cross-Encoder (ONNX) warmed up.")

    def encode(self, texts: List[str], batch_size: int = config.EMBED_BATCH_SIZE) -> Dict[str, Union[np.ndarray, List[dict]]]:
        """
        Generates hybrid embeddings for a list of texts using the hybrid ONNX model or PyTorch fallback.
        """
        if not self.bi_session and not getattr(self, "flag_model", None):
            logger.error("[EMBED] Neither bi_session nor flag_model is loaded. Cannot encode.")
            return {"dense": np.array([]), "sparse": []}

        if not texts:
            return {"dense": np.array([]), "sparse": []}

        if not hasattr(self, '_str_cache'):
            self._str_cache = {}
            
        if len(self._str_cache) > 500000:
            self._str_cache.clear()

        missing_texts = []
        missing_set = set()
        for text in texts:
            if text not in self._str_cache and text not in missing_set:
                missing_texts.append(text)
                missing_set.add(text)

        if missing_texts:
            all_dense = []
            all_sparse = []
            
            if self.bi_session:
                for i in range(0, len(missing_texts), batch_size):
                    batch = missing_texts[i : i + batch_size]
                    inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="np")
                    ort_inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
                    
                    # Run hybrid ONNX model
                    outputs = self.bi_session.run(None, ort_inputs)
                    dense_vecs = outputs[0]
                    sparse_weights = outputs[1]
                    
                    # Dense post-processing (L2 norm)
                    norms = np.linalg.norm(dense_vecs, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1e-12, norms)
                    all_dense.append(dense_vecs / norms)
                    
                    # Sparse post-processing
                    for b in range(len(batch)):
                        d = {}
                        for tok_id, w in zip(inputs["input_ids"][b], sparse_weights[b]):
                            if w > 0:
                                tok_id = int(tok_id)
                                d[tok_id] = max(d.get(tok_id, 0.0), float(w))
                        all_sparse.append(d)
            elif self.flag_model:
                # PyTorch FlagEmbedding fallback
                for i in range(0, len(missing_texts), batch_size):
                    batch = missing_texts[i : i + batch_size]
                    out = self.flag_model.encode(batch, return_dense=True, return_sparse=True)
                    all_dense.append(out["dense_vecs"])
                    for lex_dict in out["lexical_weights"]:
                        all_sparse.append({int(k): float(v) for k, v in lex_dict.items()})

            flat_dense = np.vstack(all_dense) if all_dense else np.array([])
            for i, text in enumerate(missing_texts):
                self._str_cache[text] = (flat_dense[i], all_sparse[i])

        dim = self._str_cache[texts[0]][0].shape[0]
        out_dense = np.empty((len(texts), dim), dtype=np.float32)
        out_sparse = []
        
        for i, text in enumerate(texts):
            d, s = self._str_cache[text]
            out_dense[i] = d
            out_sparse.append(s)

        return {
            "dense": out_dense,
            "sparse": out_sparse,
        }

    def encode_query(self, text: str) -> Dict[str, Union[np.ndarray, List[dict]]]:
        """Encodes a single query string."""
        return self.encode([text], batch_size=1)

    def score_cross_encoder(self, pairs: List[List[str]], batch_size: int = 32) -> np.ndarray:
        """Scores candidate pairs using the Cross-Encoder reranker (ONNX) in batches."""
        if not self.cross_session:
            return self.cross_encoder_fallback.predict(pairs, show_progress_bar=False)

        if not pairs:
            return np.array([], dtype=np.float32)

        scores = []
        iterator = tqdm(range(0, len(pairs), batch_size), desc=f"Cross-Encoder (Total Pairs: {len(pairs)})")
        for i in iterator:
            batch = pairs[i : i + batch_size]
            texts1 = [p[0] for p in batch]
            texts2 = [p[1] for p in batch]
            inputs = self.rerank_tokenizer(texts1, texts2, return_tensors="np", padding=True, truncation=True)
            ort_inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
            logits = self.cross_session.run(None, ort_inputs)[0]
            scores.extend(logits.flatten().tolist())
        return np.array(scores, dtype=np.float32)

    def check_semantic_similarity(self, text1: str, text2: str) -> float:
        """Computes cosine similarity between two strings."""
        if not text1 or not text2:
            return 0.0
        vecs = self.encode([text1, text2])
        v1, v2 = vecs["dense"]
        score = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        return score

    def embed_queries(self, queries: List[str], domain: Optional[str] = None, cache_key: str = "sku_queries") -> Dict[str, Union[np.ndarray, List[dict]]]:
        """Batch-encodes SKU queries with logging for classifier tasks."""
        if not queries:
            return {"dense": np.array([]), "sparse": []}
        logger.info(f"[EMBED] Encoding {len(queries)} SKU queries...")
        result = self.encode(queries)
        logger.info(f"[EMBED] Done encoding {len(queries)} queries.")
        return result

    def embed_weighted_sku(
        self,
        names: List[str],
        descriptions: List[str],
        categories: List[str],
        weights: tuple = (1.0, 0.8, 0.5),
    ) -> Dict[str, Union[np.ndarray, List[dict]]]:
        """
        Embeds SKUs using weighted vector averaging across Name, Description, and Category.

        Each field is embedded independently, then combined:
            final_dense = normalize(w_name * V_name + w_desc * V_desc + w_cat * V_cat)
            final_sparse = merge token dicts with weighted scores (max per token ID)

        Fields that are empty/NaN are given zero weight (skipped entirely).
        This ensures training and inference see geometrically consistent vectors
        regardless of which metadata fields are available.
        """
        if not names:
            return {"dense": np.array([]), "sparse": []}

        n = len(names)
        w_name, w_desc, w_cat = weights

        # Sanitize inputs — replace NaN/None with empty strings
        def _clean(lst):
            return [str(v).strip() if v and str(v).strip().lower() != "nan" else "" for v in lst]

        names_clean = _clean(names)
        descs_clean = _clean(descriptions) if descriptions else [""] * n
        cats_clean = _clean(categories) if categories else [""] * n

        # Embed all non-empty field lists in a single merged call to self.encode
        logger.info(f"[EMBED] Weighted SKU encoding: {n} items (weights: name={w_name}, desc={w_desc}, cat={w_cat})")

        flat_texts = []
        name_offsets = []
        desc_offsets = []
        cat_offsets = []

        for i in range(n):
            name_offsets.append(len(flat_texts))
            flat_texts.append(names_clean[i])

        has_desc = any(descs_clean)
        if has_desc:
            for i in range(n):
                if descs_clean[i]:
                    desc_offsets.append(len(flat_texts))
                    flat_texts.append(descs_clean[i])
                else:
                    desc_offsets.append(None)
        else:
            desc_offsets = [None] * n

        has_cat = any(cats_clean)
        if has_cat:
            for i in range(n):
                if cats_clean[i]:
                    cat_offsets.append(len(flat_texts))
                    flat_texts.append(cats_clean[i])
                else:
                    cat_offsets.append(None)
        else:
            cat_offsets = [None] * n

        # Single batch encoding call
        encoded = self.encode(flat_texts)
        flat_dense = encoded["dense"]
        flat_sparse = encoded["sparse"]

        # Combine dense vectors with weighted averaging
        all_dense = []
        all_sparse = []

        for i in range(n):
            name_idx = name_offsets[i]
            weighted_vec = w_name * flat_dense[name_idx]
            effective_weight = w_name

            desc_idx = desc_offsets[i]
            if desc_idx is not None:
                weighted_vec = weighted_vec + w_desc * flat_dense[desc_idx]
                effective_weight += w_desc

            cat_idx = cat_offsets[i]
            if cat_idx is not None:
                weighted_vec = weighted_vec + w_cat * flat_dense[cat_idx]
                effective_weight += w_cat

            # L2 normalize
            norm = np.linalg.norm(weighted_vec)
            if norm > 0:
                weighted_vec = weighted_vec / norm
            all_dense.append(weighted_vec)

            # Sparse: merge token dicts with weighted scores, take max per token ID
            merged_sparse = {}
            for tok_id, score in flat_sparse[name_idx].items():
                merged_sparse[tok_id] = max(merged_sparse.get(tok_id, 0.0), w_name * score)

            if desc_idx is not None:
                for tok_id, score in flat_sparse[desc_idx].items():
                    merged_sparse[tok_id] = max(merged_sparse.get(tok_id, 0.0), w_desc * score)

            if cat_idx is not None:
                for tok_id, score in flat_sparse[cat_idx].items():
                    merged_sparse[tok_id] = max(merged_sparse.get(tok_id, 0.0), w_cat * score)

            all_sparse.append(merged_sparse)

        logger.info(f"[EMBED] Weighted SKU encoding complete ({n} items).")
        return {"dense": np.vstack(all_dense), "sparse": all_sparse}

    def embed_dictionary_incremental(self, domain: str, dict_key: str, keywords: List[str]) -> Dict[str, Union[np.ndarray, List[dict]]]:
        """
        Encodes a dictionary of keywords incrementally, using disk caching to skip unchanged items.
        """
        cache_path = os.path.join(config.CACHE_DIR, f"{domain}_{dict_key}_incremental.pkl")
        os.makedirs(config.CACHE_DIR, exist_ok=True)

        item_cache = {}
        if os.path.exists(cache_path):
            try:
                item_cache = joblib.load(cache_path)
            except Exception:
                item_cache = {}

        missing_keywords = [kw for kw in keywords if kw not in item_cache]

        if missing_keywords:
            logger.info(
                f"[EMBED] {domain}/{dict_key}: {len(missing_keywords)} new items to encode "
                f"({len(keywords) - len(missing_keywords)} cached)."
            )
            all_dense, all_sparse = [], []
            batch_size = config.EMBED_BATCH_SIZE
            batches = [missing_keywords[i : i + batch_size] for i in range(0, len(missing_keywords), batch_size)]
            num_batches = len(batches)
            for idx, batch in enumerate(batches, 1):
                new_embs = self.encode(batch)
                all_dense.extend(new_embs["dense"])
                all_sparse.extend(new_embs["sparse"])
                if idx % 5 == 0 or idx == num_batches:
                    logger.info(f"[EMBED] {domain}/{dict_key}: Batch {idx}/{num_batches} encoded ({min(idx * batch_size, len(missing_keywords))}/{len(missing_keywords)} items)...")
                    sys.stdout.flush()


            for i, kw in enumerate(missing_keywords):
                item_cache[kw] = {"dense": all_dense[i], "sparse": all_sparse[i]}

            joblib.dump(item_cache, cache_path)
            logger.info(f"[EMBED] {domain}/{dict_key}: cache updated ({len(item_cache)} total items).")
        else:
            logger.info(f"[EMBED] {domain}/{dict_key}: all {len(keywords)} items served from cache.")

        dense_list = [item_cache[kw]["dense"] for kw in keywords]
        sparse_list = [item_cache[kw]["sparse"] for kw in keywords]

        return {"dense": np.array(dense_list), "sparse": sparse_list}
