import json
import logging
import os
import re
import sys
import threading
import zlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import joblib
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from engine import config
from engine.core.observability import ONNX_INFERENCE_SECONDS

logger = logging.getLogger("matchops.embedder")

_BM25_TOKEN_RE = re.compile(r"[\w\u0D80-\u0DFF\u0B80-\u0BFF]+", re.UNICODE)


class EmbeddingEngine:
    """
    Hybrid ONNX INT8 embedding and reranking engine.

    Dense vectors  -> INT8 ONNX Runtime (cls / mean / last_token pooling from model_manifest.json)
    BM25 vectors   -> Deterministic BM25 TF weights paired with Qdrant Modifier.IDF
    Sparse vectors -> Learned sparse token weights ONLY when EMBEDDING_MODEL has a sparse head (e.g. BGE-M3)
    Reranker       -> INT8 ONNX Runtime (listwise LBNL cosine or sigmoid-calibrated cross-encoder logits)
    """

    def __init__(self):
        self._cache_lock = threading.RLock()
        self._str_cache: OrderedDict[
            str, tuple[np.ndarray, dict[int, float], dict[int, float] | None]
        ] = OrderedDict()
        self._load_models()

    def _load_models(self):
        """Loads active Embedding and Reranker INT8 ONNX models strictly with fail-fast validation."""

        def _create_sess_options() -> ort.SessionOptions:
            opts = ort.SessionOptions()
            opts.add_session_config_entry("session.use_mmap_for_weights", "1")
            opts.enable_cpu_mem_arena = False
            opts.intra_op_num_threads = config.MAX_CPU_CORES
            opts.inter_op_num_threads = 1
            return opts

        def _read_manifest(path: str) -> dict:
            if os.path.exists(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                        return data if isinstance(data, dict) else {}
                except Exception:
                    pass
            return {}

        def _load_embedding():
            emb_onnx = config.EMBEDDING_ONNX
            if not os.path.exists(emb_onnx):
                raise FileNotFoundError(
                    f"[FAIL-FAST] Required INT8 ONNX embedding model missing at '{emb_onnx}'. "
                    f"Run 'python -m scripts.sync --models sync --model-target embedding' first."
                )

            logger.info(
                f"[EMBED] Loading Embedding ({config.EMBEDDING_MODEL}) INT8 ONNX from {emb_onnx}..."
            )
            self.emb_manifest = _read_manifest(config.EMBEDDING_MANIFEST)
            manifest_pooling = str(self.emb_manifest.get("pooling", "last_token")).lower()
            env_pooling = str(getattr(config, "EMBEDDING_POOLING", "auto")).lower()
            self._emb_pooling = (
                env_pooling if env_pooling in ("cls", "mean", "last_token") else manifest_pooling
            )

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    os.path.dirname(emb_onnx),
                    local_files_only=True,
                    trust_remote_code=True,
                    fix_mistral_regex=True,
                )
                if getattr(self.tokenizer, "padding_side", None) == "left":
                    self.tokenizer.padding_side = "right"
                self.bi_session = ort.InferenceSession(
                    emb_onnx,
                    _create_sess_options(),
                    providers=["CPUExecutionProvider"],
                )
            except Exception as e:
                raise RuntimeError(
                    f"[FAIL-FAST] Failed to load INT8 ONNX embedding model '{emb_onnx}': {e}"
                ) from e

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
                f"[EMBED] Embedding INT8 ONNX loaded (pooling={self._emb_pooling}, "
                f"bm25=true, learned_sparse={self.has_learned_sparse})."
            )

        def _load_reranker():
            rerank_onnx = config.RERANKER_ONNX
            if not os.path.exists(rerank_onnx):
                raise FileNotFoundError(
                    f"[FAIL-FAST] Required INT8 ONNX reranker model missing at '{rerank_onnx}'. "
                    f"Run 'python -m scripts.sync --models sync --model-target reranker' first."
                )

            logger.info(
                f"[EMBED] Loading Reranker ({config.RERANKER_MODEL}) INT8 ONNX from {rerank_onnx}..."
            )
            self.cross_manifest = _read_manifest(config.RERANKER_MANIFEST)
            default_arch = (
                "listwise_lbnl"
                if "jina-reranker-v3" in config.RERANKER_MODEL.lower()
                else "cross_encoder"
            )
            self._cross_arch = str(self.cross_manifest.get("arch_type", default_arch))
            self._score_activation = str(
                self.cross_manifest.get(
                    "score_activation",
                    "identity" if self._cross_arch == "listwise_lbnl" else "sigmoid",
                )
            )
            self._doc_embed_token = str(
                self.cross_manifest.get("doc_embed_token", "<|embed_token|>")
            )
            self._query_embed_token = str(
                self.cross_manifest.get("query_embed_token", "<|rerank_token|>")
            )
            self._doc_embed_token_id = int(self.cross_manifest.get("doc_embed_token_id", 151670))
            self._query_embed_token_id = int(
                self.cross_manifest.get("query_embed_token_id", 151671)
            )

            try:
                self.rerank_tokenizer = AutoTokenizer.from_pretrained(
                    os.path.dirname(rerank_onnx),
                    local_files_only=True,
                    trust_remote_code=True,
                    fix_mistral_regex=True,
                )
                self.cross_session = ort.InferenceSession(
                    rerank_onnx,
                    _create_sess_options(),
                    providers=["CPUExecutionProvider"],
                )
            except Exception as e:
                raise RuntimeError(
                    f"[FAIL-FAST] Failed to load INT8 ONNX reranker model '{rerank_onnx}': {e}"
                ) from e

            logger.info(
                f"[EMBED] Reranker INT8 ONNX loaded (arch={self._cross_arch}, "
                f"activation={self._score_activation})."
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            fut_emb = executor.submit(_load_embedding)
            fut_rerank = executor.submit(_load_reranker)
            fut_emb.result()
            fut_rerank.result()

    @staticmethod
    def _filter_session_inputs(session, ort_inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if hasattr(session, "get_inputs"):
            try:
                valid_names = {
                    inp.name
                    for inp in session.get_inputs()
                    if isinstance(getattr(inp, "name", None), str)
                }
                if valid_names:
                    return {k: v for k, v in ort_inputs.items() if k in valid_names}
            except Exception:
                pass
        return ort_inputs

    @staticmethod
    def _pool_hidden_states(
        hidden: np.ndarray, attention_mask: np.ndarray, pooling: str = "last_token"
    ) -> np.ndarray:
        """Pools 3D sequence hidden states (B, L, D) into 2D embeddings (B, D)."""
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

    @staticmethod
    def compute_bm25_tf(
        text: str, k1: float = 1.2, b: float = 0.75, avg_len: float = 8.0
    ) -> dict[int, float]:
        """
        Computes deterministic BM25 term-frequency saturation weights mapped to 31-bit CRC32 token IDs.
        Paired with Qdrant's server-side Modifier.IDF for native hybrid BM25 retrieval.
        """
        if not text or not isinstance(text, str):
            return {}
        tokens = [t for t in _BM25_TOKEN_RE.findall(text.lower()) if len(t) > 1 or t.isdigit()]
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

    def warmup(self):
        """Pre-allocates ONNX buffers."""
        if self.bi_session:
            dummy = self.tokenizer("warmup", return_tensors="np")
            inputs = self._filter_session_inputs(
                self.bi_session, {k: v.astype(np.int64) for k, v in dummy.items()}
            )
            self.bi_session.run(None, inputs)
            logger.info("[EMBED] Embedding (INT8 ONNX) warmed up.")

        if self.cross_session:
            if getattr(self, "_cross_arch", "cross_encoder") == "listwise_lbnl":
                dummy = self.rerank_tokenizer("warmup", return_tensors="np")
            else:
                dummy = self.rerank_tokenizer("warmup", "text", return_tensors="np")
            inputs = self._filter_session_inputs(
                self.cross_session, {k: v.astype(np.int64) for k, v in dummy.items()}
            )
            self.cross_session.run(None, inputs)
            logger.info("[EMBED] Reranker (INT8 ONNX) warmed up.")

    def encode(
        self, texts: list[str], batch_size: int = config.ONNX_BATCH_SIZE
    ) -> dict[str, np.ndarray | list[dict]]:
        """
        Generates hybrid representations for a list of texts:
          - dense:  L2-normalized embeddings from the active INT8 ONNX embedding model
          - bm25:   Deterministic BM25 TF dicts (always computed)
          - sparse: Learned sparse weights ONLY when the active ONNX model outputs them (BGE-M3); else []
        """
        if not texts:
            return {"dense": np.array([]), "bm25": [], "sparse": []}

        if not getattr(self, "bi_session", None):
            raise RuntimeError("[FAIL-FAST] Embedding INT8 ONNX session is not initialized.")

        cache_lock = getattr(self, "_cache_lock", None)
        if cache_lock is None:
            cache_lock = threading.RLock()
            self._cache_lock = cache_lock
        if not hasattr(self, "_str_cache"):
            self._str_cache = OrderedDict()

        resolved: dict[str, tuple[np.ndarray, dict[int, float], dict[int, float] | None]] = {}
        missing_texts = []
        missing_set = set()
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

        pooling = getattr(self, "_emb_pooling", "last_token")
        unused_tokens = (
            set(getattr(self.tokenizer, "all_special_ids", ()))
            if getattr(self, "tokenizer", None)
            else {0, 1, 2, 3}
        )

        if missing_texts:
            all_dense = []
            all_bm25: list[dict[int, float]] = []
            all_learned_sparse: list[dict[int, float] | None] = []

            for i in range(0, len(missing_texts), batch_size):
                batch = missing_texts[i : i + batch_size]
                inputs = self.tokenizer(
                    batch, padding=True, truncation=True, max_length=512, return_tensors="np"
                )
                ort_inputs = self._filter_session_inputs(
                    self.bi_session, {k: v.astype(np.int64) for k, v in inputs.items()}
                )

                with ONNX_INFERENCE_SECONDS.labels(model="embed").time():
                    outputs = self.bi_session.run(None, ort_inputs)
                dense_vecs = outputs[0]

                if dense_vecs.ndim == 3:
                    dense_vecs = self._pool_hidden_states(
                        dense_vecs, inputs["attention_mask"], pooling=pooling
                    )

                norms = np.linalg.norm(dense_vecs, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1e-12, norms)
                all_dense.append(dense_vecs / norms)

                for text_item in batch:
                    all_bm25.append(self.compute_bm25_tf(text_item))

                if len(outputs) > 1:
                    sparse_weights = outputs[1]
                    for b in range(len(batch)):
                        d: dict[int, float] = {}
                        for tok_id, w in zip(inputs["input_ids"][b], sparse_weights[b]):
                            tok_id = int(tok_id)
                            if tok_id not in unused_tokens and w > 0:
                                d[tok_id] = max(d.get(tok_id, 0.0), float(w))
                        all_learned_sparse.append(d)
                else:
                    all_learned_sparse.extend([None] * len(batch))

            flat_dense = np.vstack(all_dense) if all_dense else np.array([])
            max_cache_size = int(os.getenv("STR_CACHE_MAX_SIZE", "25000"))
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

    def _score_listwise_lbnl(
        self, pairs: list[list[str]], batch_size: int = config.ONNX_BATCH_SIZE
    ) -> np.ndarray:
        """
        Scores [query, doc] pairs via listwise Last-But-Not-Late (LBNL) INT8 ONNX inference,
        grouping candidates per query in a single forward pass.
        """
        doc_tok = getattr(self, "_doc_embed_token", "<|embed_token|>")
        query_tok = getattr(self, "_query_embed_token", "<|rerank_token|>")
        doc_token_id = int(getattr(self, "_doc_embed_token_id", 151670))
        query_token_id = int(getattr(self, "_query_embed_token_id", 151671))

        def _sanitize(text: str) -> str:
            return (
                str(text or "")
                .replace(doc_tok, "")
                .replace(query_tok, "")
                .replace("<|score_token|>", "")
                .strip()
            )

        query_groups: OrderedDict[str, OrderedDict[str, list[int]]] = OrderedDict()
        for pair_idx, (q_raw, d_raw) in enumerate(pairs):
            q_clean = _sanitize(q_raw)
            d_clean = _sanitize(d_raw)
            doc_map = query_groups.setdefault(q_clean, OrderedDict())
            doc_map.setdefault(d_clean, []).append(pair_idx)

        results = np.zeros(len(pairs), dtype=np.float32)
        block_size = max(1, int(batch_size))

        max_prompt_tokens = 8192
        sys_prefix = (
            "<|im_start|>system\n"
            "You are a search relevance expert who can determine a ranking of the passages based on how relevant they are to the query. "
            "If the query is a question, how relevant a passage is depends on how well it answers the question. "
            "If not, try to analyze the intent of the query and assess how well each passage satisfies the intent. "
            "If an instruction is provided, you should follow the instruction when determining the ranking."
            "<|im_end|>\n<|im_start|>user\n"
        )

        # shortcut: sequential per-query LBNL pass over up to block_size=32 candidates (~12-15s/query on CPU); swap to GPU EP for high-volume batching.
        for query_text, doc_map in query_groups.items():
            unique_docs = list(doc_map.keys())
            for start in range(0, len(unique_docs), block_size):
                chunk_docs = unique_docs[start : start + block_size]
                query_suffix = (
                    f"\n<query>\n{query_text}{query_tok}\n</query>"
                    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                )
                header = (
                    f"{sys_prefix}I will provide you with {len(chunk_docs)} passages, "
                    f"each indicated by a numerical identifier. "
                    f"Rank the passages based on their relevance to query: {query_text}\n"
                )

                # Budget passage tokens so the trailing <query> block cannot be right-truncated past max_prompt_tokens.
                overhead_len = len(
                    self.rerank_tokenizer(header + query_suffix, add_special_tokens=False)[
                        "input_ids"
                    ]
                )
                avail_for_docs = max(len(chunk_docs) * 32, max_prompt_tokens - overhead_len - 16)
                per_doc_max = max(24, (avail_for_docs // max(1, len(chunk_docs))) - 14)

                passages_parts = []
                for idx, doc in enumerate(chunk_docs):
                    doc_ids = self.rerank_tokenizer(doc, add_special_tokens=False)["input_ids"]
                    if len(doc_ids) > per_doc_max:
                        doc = self.rerank_tokenizer.decode(
                            doc_ids[:per_doc_max], skip_special_tokens=True
                        ).strip()
                    passages_parts.append(f'<passage id="{idx}">\n{doc}{doc_tok}\n</passage>')
                prompt = header + "\n".join(passages_parts) + query_suffix

                inputs = self.rerank_tokenizer(
                    [prompt],
                    return_tensors="np",
                    truncation=True,
                    max_length=max_prompt_tokens,
                )
                ort_inputs = self._filter_session_inputs(
                    self.cross_session, {k: v.astype(np.int64) for k, v in inputs.items()}
                )
                with ONNX_INFERENCE_SECONDS.labels(model="cross_encoder").time():
                    proj_out = self.cross_session.run(None, ort_inputs)[0]

                input_ids_row = inputs["input_ids"][0]
                doc_positions = np.where(input_ids_row == doc_token_id)[0]
                query_positions = np.where(input_ids_row == query_token_id)[0]
                if len(query_positions) == 0 or len(doc_positions) < len(chunk_docs):
                    logger.warning(
                        f"[EMBED] LBNL readout token mismatch (docs={len(doc_positions)}/{len(chunk_docs)}, "
                        f"query={len(query_positions)}); assigning 0.0 scores for block."
                    )
                    continue

                emb_seq = proj_out[0] if proj_out.ndim == 3 else proj_out
                q_vec = emb_seq[int(query_positions[-1])].astype(np.float32)
                q_norm = float(np.linalg.norm(q_vec))
                if q_norm > 0:
                    q_vec = q_vec / q_norm

                d_vecs = emb_seq[doc_positions[: len(chunk_docs)]].astype(np.float32)
                d_norms = np.linalg.norm(d_vecs, axis=1, keepdims=True)
                d_norms = np.where(d_norms == 0, 1e-12, d_norms)
                d_vecs = d_vecs / d_norms
                cos_scores = (d_vecs @ q_vec).astype(np.float32)

                for doc_idx, doc_text in enumerate(chunk_docs):
                    score_val = float(cos_scores[doc_idx])
                    for orig_idx in doc_map[doc_text]:
                        results[orig_idx] = score_val

        return results

    def score_cross_encoder(
        self, pairs: list[list[str]], batch_size: int = config.ONNX_BATCH_SIZE
    ) -> np.ndarray:
        """Scores candidate pairs using INT8 ONNX Reranker (listwise LBNL or cross-encoder)."""
        if not pairs:
            return np.array([], dtype=np.float32)

        if not self.cross_session or not self.rerank_tokenizer:
            logger.error("[EMBED] Reranker ONNX session unavailable; returning -10.0 sentinels.")
            return np.full(len(pairs), -10.0, dtype=np.float32)

        if getattr(self, "_cross_arch", "cross_encoder") == "listwise_lbnl" and hasattr(
            self.rerank_tokenizer, "encode"
        ):
            logger.debug(f"[EMBED] Scoring {len(pairs)} pairs with Listwise LBNL INT8 Reranker...")
            return self._score_listwise_lbnl(pairs, batch_size=batch_size)

        logger.debug(f"[EMBED] Scoring {len(pairs)} pairs with Cross-Encoder INT8 ONNX...")
        # Dynamic INT8 quantizes activations per-batch, so preserve exact batch composition.
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            texts1 = [p[0] for p in batch]
            texts2 = [p[1] for p in batch]
            inputs = self.rerank_tokenizer(
                texts1,
                texts2,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=512,
            )
            ort_inputs = self._filter_session_inputs(
                self.cross_session, {k: v.astype(np.int64) for k, v in inputs.items()}
            )
            with ONNX_INFERENCE_SECONDS.labels(model="cross_encoder").time():
                logits = self.cross_session.run(None, ort_inputs)[0]
            arr = np.asarray(logits, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                arr = arr[:, 1] - arr[:, 0]
            else:
                arr = arr.reshape(-1)
            scores.extend(arr.tolist())

        out = np.array(scores, dtype=np.float32)
        if getattr(self, "_score_activation", "identity") == "sigmoid":
            out = (1.0 / (1.0 + np.exp(-np.clip(out, -60.0, 60.0)))).astype(np.float32)
        return out

    @staticmethod
    def predict_centroids_temperature(
        vecs: np.ndarray,
        prototypes: np.ndarray,
        tau: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates temperature-scaled class probabilities via batch matrix multiplication.
        vecs: (N, D) or (D,) normalized query embeddings
        prototypes: (C, D) normalized centroid prototype vectors
        tau: temperature scaling factor (default: 0.05)

        Returns:
            Tuple of (probas (N, C), best_indices (N,), confidences (N,))
        """
        vecs_2d = np.atleast_2d(vecs)
        protos_2d = np.atleast_2d(prototypes)

        if vecs_2d.shape[0] == 0 or protos_2d.shape[0] == 0:
            return (
                np.empty((vecs_2d.shape[0], protos_2d.shape[0]), dtype=np.float32),
                np.empty(vecs_2d.shape[0], dtype=np.int64),
                np.empty(vecs_2d.shape[0], dtype=np.float32),
            )

        safe_tau = max(float(tau), 1e-6)
        logits = (vecs_2d @ protos_2d.T) / safe_tau

        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        sum_exp = np.sum(exp_logits, axis=-1, keepdims=True)
        sum_exp = np.where(sum_exp == 0, 1e-12, sum_exp)
        probas = (exp_logits / sum_exp).astype(np.float32)

        best_indices = np.argmax(probas, axis=-1)
        confidences = np.max(probas, axis=-1)

        return probas, best_indices, confidences

    def embed_queries(
        self, queries: list[str], domain: str | None = None, cache_key: str = "sku_queries"
    ) -> dict[str, np.ndarray | list[dict]]:
        """Batch-encodes SKU queries with logging for classifier tasks."""
        if not queries:
            return {"dense": np.array([]), "bm25": [], "sparse": []}
        logger.info(f"[EMBED] Encoding {len(queries)} SKU queries...")
        result = self.encode(queries)
        logger.info(f"[EMBED] Done encoding {len(queries)} queries.")
        return result

    @staticmethod
    def _merge_weighted_token_dicts(
        flat_dicts: list[dict[int, float]],
        name_idx: int,
        desc_idx: int | None,
        cat_idx: int | None,
        weights: tuple[float, float, float],
    ) -> dict[int, float]:
        w_name, w_desc, w_cat = weights
        merged: dict[int, float] = {}
        for idx, weight in ((name_idx, w_name), (desc_idx, w_desc), (cat_idx, w_cat)):
            if idx is None:
                continue
            for tok_id, score in flat_dicts[idx].items():
                merged[tok_id] = max(merged.get(tok_id, 0.0), weight * score)
        return merged

    def embed_weighted_sku(
        self,
        names: list[str],
        descriptions: list[str],
        categories: list[str],
        weights: tuple = config.CLASSIFIER_WEIGHTS,
    ) -> dict[str, np.ndarray | list[dict]]:
        """
        Embeds SKUs using weighted vector averaging across Name, Description, and Category.

        Each field is embedded independently, then combined:
            final_dense = normalize(w_name * V_name + w_desc * V_desc + w_cat * V_cat)
            final_bm25  = merge BM25 TF dicts with weighted scores (max per token ID)
            final_sparse = merge learned sparse dicts when available (BGE-M3 only)
        """
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

        logger.debug(
            f"[EMBED] Weighted SKU encoding: {n} items (weights: name={w_name}, desc={w_desc}, cat={w_cat})"
        )

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
        flat_bm25 = encoded.get("bm25") or [self.compute_bm25_tf(t) for t in flat_texts]
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
                self._merge_weighted_token_dicts(flat_bm25, name_idx, desc_idx, cat_idx, weights)
            )
            if has_sparse:
                all_sparse.append(
                    self._merge_weighted_token_dicts(
                        flat_sparse, name_idx, desc_idx, cat_idx, weights
                    )
                )

        logger.debug(f"[EMBED] Weighted SKU encoding complete ({n} items).")
        return {"dense": np.vstack(all_dense), "bm25": all_bm25, "sparse": all_sparse}

    def embed_dictionary_incremental(
        self, domain: str, dict_key: str, keywords: list[str]
    ) -> dict[str, np.ndarray | list[dict]]:
        """
        Encodes a dictionary of keywords incrementally, using disk caching to skip unchanged items.
        """
        cache_path = os.path.join(config.CACHE_DIR, f"{domain}_{dict_key}_incremental.pkl")
        os.makedirs(config.CACHE_DIR, exist_ok=True)

        item_cache: dict = {}
        if os.path.exists(cache_path):
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
                f"[EMBED] {domain}/{dict_key}: {len(missing_keywords)} new items to encode "
                f"({len(keywords) - len(missing_keywords)} cached)."
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
                        f"[EMBED] {domain}/{dict_key}: Batch {idx}/{num_batches} encoded "
                        f"({min(idx * batch_size, len(missing_keywords))}/{len(missing_keywords)} items)..."
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
            logger.info(
                f"[EMBED] {domain}/{dict_key}: cache updated ({len(item_cache)} total items)."
            )
        else:
            logger.info(
                f"[EMBED] {domain}/{dict_key}: all {len(keywords)} items served from cache."
            )

        dense_list = [item_cache[kw]["dense"] for kw in keywords]
        bm25_list = [item_cache[kw]["bm25"] for kw in keywords]
        sparse_list = (
            [item_cache[kw]["sparse"] for kw in keywords]
            if keywords and all("sparse" in item_cache[kw] for kw in keywords)
            else []
        )

        return {"dense": np.array(dense_list), "bm25": bm25_list, "sparse": sparse_list}
