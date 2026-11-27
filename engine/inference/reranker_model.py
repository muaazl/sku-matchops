"""ONNX Runtime cross-encoder and listwise LBNL reranker model wrapper."""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from engine.core import config
from engine.core.observability import ONNX_INFERENCE_SECONDS
from engine.core.onnx_factory import create_onnx_session, filter_onnx_inputs

logger = logging.getLogger("matchops.inference.reranker")


class RerankerModel:
    """ONNX reranker supporting listwise LBNL cosine scoring and cross-encoder logits."""

    def __init__(self) -> None:
        self._load_reranker_model()

    def _load_reranker_model(self) -> None:
        """Loads the active ONNX reranker session and tokenizer with fail-fast checks."""
        rerank_onnx = Path(config.RERANKER_ONNX)
        if not rerank_onnx.exists():
            raise FileNotFoundError(
                f"[FAIL-FAST] Required ONNX reranker model missing at '{rerank_onnx}'. "
                "Run 'python -m scripts.sync --models sync --model-target reranker' first."
            )

        prec_label = getattr(config, "RERANKER_ACTIVE_PRECISION", "int8").upper()
        logger.info(
            "[RERANK] Loading Reranker (%s) %s ONNX from %s...",
            config.RERANKER_MODEL,
            prec_label,
            rerank_onnx,
        )
        self.cross_manifest = config.read_json_manifest(config.RERANKER_MANIFEST)
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
        self._doc_embed_token = str(self.cross_manifest.get("doc_embed_token", "<|embed_token|>"))
        self._query_embed_token = str(
            self.cross_manifest.get("query_embed_token", "<|rerank_token|>")
        )
        self._doc_embed_token_id = int(self.cross_manifest.get("doc_embed_token_id", 151670))
        self._query_embed_token_id = int(self.cross_manifest.get("query_embed_token_id", 151671))

        try:
            try:
                self.rerank_tokenizer = AutoTokenizer.from_pretrained(
                    str(rerank_onnx.parent),
                    local_files_only=True,
                    trust_remote_code=True,
                    fix_mistral_regex=True,
                )
            except TypeError:
                self.rerank_tokenizer = AutoTokenizer.from_pretrained(
                    str(rerank_onnx.parent),
                    local_files_only=True,
                    trust_remote_code=True,
                    fix_mistral_regex=False,
                )
            self.cross_session, _ = create_onnx_session(
                rerank_onnx, getattr(config, "RERANKER_ORT_PROVIDERS", None)
            )
        except Exception as exc:
            raise RuntimeError(
                f"[FAIL-FAST] Failed to load ONNX reranker model '{rerank_onnx}': {exc}"
            ) from exc

        logger.info(
            "[RERANK] Reranker %s ONNX loaded (arch=%s, activation=%s).",
            prec_label,
            self._cross_arch,
            self._score_activation,
        )

    def warmup(self) -> None:
        """Runs a tiny dummy forward pass to pre-allocate ONNX Runtime buffers."""
        if getattr(self, "cross_session", None):
            if getattr(self, "_cross_arch", "cross_encoder") == "listwise_lbnl":
                dummy = self.rerank_tokenizer("warmup", return_tensors="np")
            else:
                dummy = self.rerank_tokenizer("warmup", "text", return_tensors="np")
            inputs = filter_onnx_inputs(
                self.cross_session, {k: v.astype(np.int64) for k, v in dummy.items()}
            )
            self.cross_session.run(None, inputs)
            logger.info("[RERANK] Reranker (ONNX) warmed up.")

    def _score_listwise_lbnl(
        self,
        pairs: list[list[str]],
        batch_size: int = config.ONNX_BATCH_SIZE,
        progress_callback=None,
    ) -> np.ndarray:
        """Scores `[query, doc]` pairs via listwise Last-But-Not-Late (LBNL) ONNX inference."""
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
        total_blocks = sum((len(dm) + block_size - 1) // block_size for dm in query_groups.values())
        done_blocks = 0

        max_prompt_tokens = 8192
        sys_prefix = (
            "<|im_start|>system\n"
            "You are a search relevance expert who can determine a ranking of the passages based on how relevant they are to the query. "
            "If the query is a question, how relevant a passage is depends on how well it answers the question. "
            "If not, try to analyze the intent of the query and assess how well each passage satisfies the intent. "
            "If an instruction is provided, you should follow the instruction when determining the ranking."
            "<|im_end|>\n<|im_start|>user\n"
        )

        if progress_callback and total_blocks > 0:
            progress_callback(0, total_blocks)

        # shortcut: sequential per-query LBNL pass over up to block_size=32 candidates (~12-15s/query on CPU); swap to GPU EP for high-volume batching.
        for query_text, doc_map in query_groups.items():
            unique_docs = sorted(doc_map.keys())
            for start in range(0, len(unique_docs), block_size):
                chunk_docs = unique_docs[start : start + block_size]
                query_suffix = (
                    f"\n<query>\n{query_text}{query_tok}\n</query>"
                    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                )
                header = (
                    f"{sys_prefix}I will provide you with {len(chunk_docs)} passages, "
                    "each indicated by a numerical identifier. "
                    f"Rank the passages based on their relevance to query: {query_text}\n"
                )

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
                ort_inputs = filter_onnx_inputs(
                    self.cross_session, {k: v.astype(np.int64) for k, v in inputs.items()}
                )
                with ONNX_INFERENCE_SECONDS.labels(model="cross_encoder").time():
                    proj_out = self.cross_session.run(None, ort_inputs)[0]

                input_ids_row = inputs["input_ids"][0]
                doc_positions = np.nonzero(input_ids_row == doc_token_id)[0]
                query_positions = np.nonzero(input_ids_row == query_token_id)[0]
                if len(query_positions) == 0 or len(doc_positions) < len(chunk_docs):
                    logger.warning(
                        "[RERANK] LBNL readout token mismatch (docs=%d/%d, query=%d); assigning 0.0 scores for block.",
                        len(doc_positions),
                        len(chunk_docs),
                        len(query_positions),
                    )
                    del proj_out, ort_inputs, inputs
                    done_blocks += 1
                    if progress_callback:
                        progress_callback(done_blocks, total_blocks)
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
                del proj_out, emb_seq, ort_inputs, inputs

                for doc_idx, doc_text in enumerate(chunk_docs):
                    score_val = float(cos_scores[doc_idx])
                    for orig_idx in doc_map[doc_text]:
                        results[orig_idx] = score_val

                done_blocks += 1
                if progress_callback:
                    progress_callback(done_blocks, total_blocks)

        return results

    def score_cross_encoder(
        self,
        pairs: list[list[str]],
        batch_size: int = config.ONNX_BATCH_SIZE,
        progress_callback=None,
    ) -> np.ndarray:
        """Scores candidate pairs using ONNX Reranker (listwise LBNL or cross-encoder)."""
        if not pairs:
            return np.array([], dtype=np.float32)

        if not getattr(self, "cross_session", None) or not getattr(self, "rerank_tokenizer", None):
            logger.error("[RERANK] Reranker ONNX session unavailable; returning -10.0 sentinels.")
            return np.full(len(pairs), -10.0, dtype=np.float32)

        if getattr(self, "_cross_arch", "cross_encoder") == "listwise_lbnl" and hasattr(
            self.rerank_tokenizer, "encode"
        ):
            return self._score_listwise_lbnl(
                pairs, batch_size=batch_size, progress_callback=progress_callback
            )

        # Preserve exact batch composition for dynamic INT8 activation quantization.
        scores = []
        total_batches = (len(pairs) + batch_size - 1) // batch_size
        if progress_callback and total_batches > 0:
            progress_callback(0, total_batches)
        for batch_idx, i in enumerate(range(0, len(pairs), batch_size), 1):
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
            ort_inputs = filter_onnx_inputs(
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
            if progress_callback:
                progress_callback(batch_idx, total_batches)

        out = np.array(scores, dtype=np.float32)
        if getattr(self, "_score_activation", "identity") == "sigmoid":
            out = (1.0 / (1.0 + np.exp(-np.clip(out, -60.0, 60.0)))).astype(np.float32)
        return out

    def predict(self, pairs: list[list[str]], progress_callback=None) -> np.ndarray:
        """Scores pairs with optional progress callback compatibility."""
        return self.score_cross_encoder(pairs, progress_callback=progress_callback)
