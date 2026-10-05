import enum
import logging
import os
import sqlite3
import threading
from typing import Any

import joblib
import numpy as np
import pandas as pd

from engine import config
from engine.nlp.embedding_engine import EmbeddingEngine

logger = logging.getLogger("matchops.cold_start_router")


class ClassLifecycleTier(enum.Enum):
    """Lifecycle tier of a Basic Type category based on historical manual examples N_c."""

    TIER_1_ZERO_SHOT = 1  # N_c == 0 (Pure Zero-Shot Cross-Encoder)
    TIER_2_FEW_SHOT = 2  # 1 <= N_c < 15 (Cosine-Weighted k-NN + ML-kNN)
    TIER_3_WARM_CENTROID = 3  # N_c >= 15 (Dense Prototypes / Matrix Multiplication)


class TaxonomyLifecycleRegistry:
    """
    Thread-safe in-memory registry of class lifecycle states (N_c) and precomputed prototypes.
    Eliminates per-inference Qdrant count latency by maintaining taxonomy counts in memory.
    """

    def __init__(
        self,
        domain: str = config.DOMAIN_MARKET,
        cache_dir: str | None = None,
        few_shot_threshold: int = config.LIFECYCLE_FEW_SHOT_THRESHOLD,
    ):
        self.domain = domain
        self.cache_dir = cache_dir or config.CACHE_DIR
        self.few_shot_threshold = few_shot_threshold
        self._lock = threading.RLock()

        # Class counts N_c and lifecycle tier assignment
        self.class_counts: dict[str, int] = {}
        self.class_tiers: dict[str, ClassLifecycleTier] = {}
        self.class_descriptions: dict[str, str] = {}

        # Partitioned class sets
        self.zero_shot_classes: list[str] = []
        self.few_shot_classes: list[str] = []
        self.warm_classes: list[str] = []

        # Warm Centroids (shape: M_warm x 1024, L2 normalized)
        self.warm_centroids: np.ndarray = np.empty((0, 1024), dtype=np.float32)
        self.warm_index_to_class: list[str] = []
        self.class_to_warm_index: dict[str, int] = {}

        # Zero-Shot bi-encoder description embeddings for fast pre-filtering (M_zero x 1024)
        self.zero_shot_embeddings: np.ndarray = np.empty((0, 1024), dtype=np.float32)
        self.zero_shot_index_to_class: list[str] = []

    def get_tier(self, class_name: str) -> ClassLifecycleTier:
        with self._lock:
            return self.class_tiers.get(class_name, ClassLifecycleTier.TIER_1_ZERO_SHOT)

    def get_count(self, class_name: str) -> int:
        with self._lock:
            return self.class_counts.get(class_name, 0)

    def register_tag(
        self,
        tag: str,
        description: str = "",
        sample_count: int = 0,
        embed_engine: EmbeddingEngine | None = None,
    ):
        """
        Dynamically registers a new Basic Type tag into the lifecycle registry.
        Guaranteed thread-safe and updates tier partitioning without runtime Qdrant queries.
        """
        clean_tag = tag.strip()
        if not clean_tag:
            return

        with self._lock:
            self.class_counts[clean_tag] = sample_count
            if description:
                self.class_descriptions[clean_tag] = description
            elif clean_tag not in self.class_descriptions:
                self.class_descriptions[clean_tag] = clean_tag

            self._partition_classes()

            # If it's a zero-shot tag and embed_engine is provided, compute its description embedding
            if (
                self.class_tiers[clean_tag] == ClassLifecycleTier.TIER_1_ZERO_SHOT
                and embed_engine is not None
            ):
                self._update_single_zero_shot_embedding(clean_tag, embed_engine)

    def _partition_classes(self):
        """Re-partitions all registered classes into zero-shot, few-shot, and warm tiers."""
        self.zero_shot_classes = []
        self.few_shot_classes = []
        self.warm_classes = []

        for c, count in self.class_counts.items():
            if count == 0:
                self.class_tiers[c] = ClassLifecycleTier.TIER_1_ZERO_SHOT
                self.zero_shot_classes.append(c)
            elif 1 <= count < self.few_shot_threshold:
                self.class_tiers[c] = ClassLifecycleTier.TIER_2_FEW_SHOT
                self.few_shot_classes.append(c)
            else:
                self.class_tiers[c] = ClassLifecycleTier.TIER_3_WARM_CENTROID
                self.warm_classes.append(c)

        self.zero_shot_classes.sort()
        self.few_shot_classes.sort()
        self.warm_classes.sort()

    def _update_single_zero_shot_embedding(self, tag: str, embed_engine: EmbeddingEngine):
        """Appends or updates the bi-encoder embedding for a single zero-shot class."""
        desc_text = self.class_descriptions.get(tag, tag)
        formatted_prompt = (
            f"Basic Type: {tag} - {desc_text}"
            if desc_text and desc_text != tag
            else f"Basic Type: {tag}"
        )
        res = embed_engine.encode([formatted_prompt])
        vec = res["dense"][0].astype(np.float32)

        if tag in self.zero_shot_index_to_class:
            idx = self.zero_shot_index_to_class.index(tag)
            self.zero_shot_embeddings[idx] = vec
        else:
            self.zero_shot_index_to_class.append(tag)
            if self.zero_shot_embeddings.shape[0] == 0:
                self.zero_shot_embeddings = vec.reshape(1, -1)
            else:
                self.zero_shot_embeddings = np.vstack([self.zero_shot_embeddings, vec])

    def sync_from_catalog(
        self,
        registered_bts: list[str],
        descriptions: dict[str, str],
        cat_df: pd.DataFrame | None = None,
        embed_engine: EmbeddingEngine | None = None,
        force_rebuild: bool = False,
    ):
        """
        Initializes the taxonomy lifecycle registry from catalog counts and precomputed prototypes.
        """
        with self._lock:
            # 1. Update descriptions
            for bt in registered_bts:
                clean_bt = bt.strip()
                if clean_bt:
                    self.class_descriptions[clean_bt] = descriptions.get(clean_bt, clean_bt)

            # 2. Derive class counts N_c
            counts: dict[str, int] = {}
            if cat_df is not None and not cat_df.empty:
                bt_col = None
                for col in ["basictype", "BasicType", "basic_type", "BT"]:
                    if col in cat_df.columns:
                        bt_col = col
                        break
                if bt_col:
                    vc = cat_df[bt_col].astype(str).str.strip().value_counts()
                    counts = {str(k): int(v) for k, v in vc.items() if str(k).strip()}
            elif os.path.exists(config.DB_PATH):
                # Fall back to SQLite catalog_items table
                try:
                    conn = sqlite3.connect(config.DB_PATH)
                    cur = conn.cursor()
                    cur.execute(
                        "SELECT basictype, COUNT(*) FROM catalog_items WHERE domain = ? GROUP BY basictype",
                        (self.domain,),
                    )
                    rows = cur.fetchall()
                    conn.close()
                    counts = {
                        str(r[0]).strip(): int(r[1]) for r in rows if r[0] and str(r[0]).strip()
                    }
                except Exception as db_err:
                    logger.warning(
                        f"[{self.domain}] Could not read class counts from SQLite: {db_err}"
                    )

            # Assign counts for all known BTs
            self.class_counts = {bt: counts.get(bt, 0) for bt in self.class_descriptions.keys()}
            # Include any catalog BTs not yet in registered_bts
            for bt, cnt in counts.items():
                if bt not in self.class_counts:
                    self.class_counts[bt] = cnt
                    if bt not in self.class_descriptions:
                        self.class_descriptions[bt] = bt

            # 3. Partition into Tiers
            self._partition_classes()
            logger.info(
                f"[TAXONOMY] [{self.domain.upper()}] Partitioned {len(self.class_counts)} classes: "
                f"{len(self.zero_shot_classes)} Zero-Shot (N_c=0) | "
                f"{len(self.few_shot_classes)} Few-Shot (1<=N_c<15) | "
                f"{len(self.warm_classes)} Warm Centroids (N_c>=15)"
            )

            # 4. Build or load Warm Centroids (instantaneous via cached embeddings or warm_centroids.pkl)
            self._build_or_load_centroids(cat_df, embed_engine, force_rebuild)

    def _build_or_load_centroids(
        self,
        cat_df: pd.DataFrame | None,
        embed_engine: EmbeddingEngine | None,
        force_rebuild: bool,
    ):
        """Precomputes or loads L2-normalized centroid vectors for all warm classes (N_c >= 15)."""
        cache_path = os.path.join(self.cache_dir, f"{self.domain}_warm_centroids.pkl")
        os.makedirs(self.cache_dir, exist_ok=True)

        if not force_rebuild and os.path.exists(cache_path):
            try:
                cached = joblib.load(cache_path)
                cached_classes = cached.get("classes", [])
                # Verify that warm classes match cached classes
                if set(cached_classes) == set(self.warm_classes):
                    self.warm_centroids = cached["centroids"]
                    self.warm_index_to_class = cached_classes
                    self.class_to_warm_index = {
                        c: i for i, c in enumerate(self.warm_index_to_class)
                    }
                    logger.info(
                        f"[CENTROIDS] Loaded {len(self.warm_index_to_class)} warm centroids from cache ({self.domain})."
                    )
                    return
            except Exception as e:
                logger.warning(
                    f"[CENTROIDS] Cache read failed for {self.domain} ({e}); recomputing."
                )

        if cat_df is None or cat_df.empty:
            logger.debug(
                f"[CENTROIDS] No cat_df available for {self.domain} centroid precomputation."
            )
            return

        bt_col = None
        for col in ["basictype", "BasicType", "basic_type", "BT"]:
            if col in cat_df.columns:
                bt_col = col
                break
        if not bt_col or not self.warm_classes:
            return

        df_clean = cat_df.dropna(subset=[bt_col]).copy()
        df_clean[bt_col] = df_clean[bt_col].astype(str).str.strip()

        names = (
            df_clean["Name"].astype(str).str.strip().tolist()
            if "Name" in df_clean.columns
            else [""] * len(df_clean)
        )
        descs = (
            df_clean["Description"].astype(str).str.strip().tolist()
            if "Description" in df_clean.columns
            else [""] * len(df_clean)
        )
        col_cat = (
            "category"
            if "category" in df_clean.columns
            else ("Category" if "Category" in df_clean.columns else "")
        )
        cats = (
            df_clean[col_cat].astype(str).str.strip().tolist() if col_cat else [""] * len(df_clean)
        )

        centroids_list = []
        valid_warm_classes = []

        # Fast path: leverage already computed weighted_skus_cache without invoking neural net
        weighted_cache_path = os.path.join(self.cache_dir, f"{self.domain}_weighted_skus_cache.pkl")
        if os.path.exists(weighted_cache_path):
            try:
                sku_cache = joblib.load(weighted_cache_path)
                keys = [
                    f"{str(n).strip()}||{str(d).strip()}||{str(c).strip()}"
                    for n, d, c in zip(names, descs, cats)
                ]
                labels = df_clean[bt_col].values

                # Group by warm class and calculate centroids directly
                for c in self.warm_classes:
                    cls_indices = np.where(labels == c)[0]
                    cls_vecs = [
                        sku_cache[keys[idx]] for idx in cls_indices if keys[idx] in sku_cache
                    ]
                    if not cls_vecs:
                        continue
                    class_matrix = np.vstack(cls_vecs)
                    raw_centroid = np.mean(class_matrix, axis=0)
                    norm = np.linalg.norm(raw_centroid)
                    normed_centroid = (raw_centroid / norm) if norm > 0 else raw_centroid
                    centroids_list.append(normed_centroid)
                    valid_warm_classes.append(c)

                logger.info(
                    f"[CENTROIDS] Precomputed {len(valid_warm_classes)} centroids from cached SKU embeddings ({self.domain})."
                )
            except Exception as cache_err:
                logger.warning(
                    f"[CENTROIDS] Error reading weighted_skus_cache ({cache_err}); skipping."
                )

        if centroids_list:
            self.warm_centroids = np.vstack(centroids_list).astype(np.float32)
            self.warm_index_to_class = valid_warm_classes
            self.class_to_warm_index = {c: i for i, c in enumerate(self.warm_index_to_class)}

            joblib.dump(
                {"classes": self.warm_index_to_class, "centroids": self.warm_centroids},
                cache_path,
            )
            logger.info(f"[CENTROIDS] Successfully cached {len(self.warm_centroids)} prototypes.")

    def ensure_zero_shot_embeddings(self, embed_engine: EmbeddingEngine):
        """Lazily precomputes bi-encoder embeddings for zero-shot class descriptions when queried."""
        with self._lock:
            if not self.zero_shot_classes:
                self.zero_shot_embeddings = np.empty((0, 1024), dtype=np.float32)
                self.zero_shot_index_to_class = []
                return

            if self.zero_shot_embeddings.shape[0] == len(self.zero_shot_classes):
                return

            prompts = []
            for c in self.zero_shot_classes:
                desc = self.class_descriptions.get(c, "")
                prompt = f"Basic Type: {c} - {desc}" if desc and desc != c else f"Basic Type: {c}"
                prompts.append(prompt)

            res = embed_engine.encode(prompts)
            self.zero_shot_embeddings = res["dense"].astype(np.float32)
            self.zero_shot_index_to_class = list(self.zero_shot_classes)


class Tier1ZeroShotClassifier:
    """
    Tier 1: Pure Zero-Shot ($N_c = 0$ examples in Qdrant).
    Method: Cross-Encoder Text Classification using BGE-Reranker-v2-m3.
    Formats pairs: (sku_name, "Basic Type: {bt_name} - {bt_description}").
    """

    def __init__(
        self,
        embed_engine: EmbeddingEngine,
        registry: TaxonomyLifecycleRegistry,
        confidence_threshold: float = config.REVIEW_THRESHOLD,
        max_candidate_cross_eval: int = config.ZERO_SHOT_MAX_CANDIDATES,
    ):
        self.embed_engine = embed_engine
        self.registry = registry
        self.confidence_threshold = confidence_threshold
        self.max_candidate_cross_eval = max_candidate_cross_eval

    def predict(
        self,
        sku_name: str,
        sku_description: str = "",
        query_dense: np.ndarray | None = None,
    ) -> tuple[str, float, str]:
        """
        Evaluates an SKU against zero-shot classes.
        Returns: (bt_label, confidence, source).
        """
        zero_shot_classes = self.registry.zero_shot_classes
        if not zero_shot_classes:
            return "", 0.0, "zero-shot"

        # 1. Candidate Selection: if many zero-shot classes, pre-filter with bi-encoder
        candidate_classes = zero_shot_classes
        if len(zero_shot_classes) > self.max_candidate_cross_eval and query_dense is not None:
            self.registry.ensure_zero_shot_embeddings(self.embed_engine)
            zs_embs = self.registry.zero_shot_embeddings
            if zs_embs.shape[0] == len(zero_shot_classes):
                q_vec = query_dense.reshape(1, -1)
                sims = (q_vec @ zs_embs.T)[0]
                top_k_indices = np.argsort(sims)[::-1][: self.max_candidate_cross_eval]
                candidate_classes = [
                    self.registry.zero_shot_index_to_class[i] for i in top_k_indices
                ]

        # 2. Format cross-encoder pairs: (sku_name, "Basic Type: {bt_name} - {bt_description}")
        query_text = (
            f"{sku_name} {sku_description}".strip() if sku_description else sku_name.strip()
        )
        pairs = []
        for bt in candidate_classes:
            desc = self.registry.class_descriptions.get(bt, "")
            bt_str = f"Basic Type: {bt} - {desc}" if desc and desc != bt else f"Basic Type: {bt}"
            pairs.append([query_text, bt_str])

        # 3. Score with cross-encoder
        raw_logits = self.embed_engine.score_cross_encoder(pairs)
        if len(raw_logits) == 0:
            return "", 0.0, "zero-shot"

        # 4. Calibrate confidence: sigmoid(logit)
        confs = 1.0 / (1.0 + np.exp(-raw_logits))
        best_idx = int(np.argmax(confs))
        best_conf = float(confs[best_idx])
        best_class = candidate_classes[best_idx]

        if best_conf >= self.confidence_threshold:
            return best_class, best_conf, "zero-shot"
        return "", 0.0, "zero-shot"

    def batch_predict(
        self,
        sku_names: list[str],
        sku_descriptions: list[str] | None = None,
        query_dense_matrix: np.ndarray | None = None,
    ) -> list[tuple[str, float, str]]:
        """Batch version of Tier 1 Zero-Shot prediction with fully vectorized Cross-Encoder inference."""
        n = len(sku_names)
        if n == 0:
            return []

        if sku_descriptions is None:
            sku_descriptions = [""] * n

        zero_shot_classes = self.registry.zero_shot_classes
        if not zero_shot_classes:
            return [("", 0.0, "zero-shot")] * n

        # 1. Candidate Selection
        candidate_classes_per_sku = [zero_shot_classes] * n
        if (
            len(zero_shot_classes) > self.max_candidate_cross_eval
            and query_dense_matrix is not None
        ):
            self.registry.ensure_zero_shot_embeddings(self.embed_engine)
            zs_embs = self.registry.zero_shot_embeddings
            if zs_embs.shape[0] == len(zero_shot_classes):
                # Matrix multiplication: (n, 1024) @ (1024, M_zero) -> (n, M_zero)
                sims_matrix = query_dense_matrix @ zs_embs.T
                for i in range(n):
                    top_k_indices = np.argsort(sims_matrix[i])[::-1][
                        : self.max_candidate_cross_eval
                    ]
                    candidate_classes_per_sku[i] = [
                        self.registry.zero_shot_index_to_class[idx] for idx in top_k_indices
                    ]

        # 2. Format cross-encoder pairs for all SKUs
        all_pairs = []
        flat_meta = []  # Tuples of (sku_idx, candidate_class_name)

        for i in range(n):
            if not sku_names[i]:
                continue
            name_txt = sku_names[i]
            desc_txt = sku_descriptions[i]
            query_text = f"{name_txt} {desc_txt}".strip() if desc_txt else name_txt.strip()

            for bt in candidate_classes_per_sku[i]:
                desc = self.registry.class_descriptions.get(bt, "")
                bt_str = (
                    f"Basic Type: {bt} - {desc}" if desc and desc != bt else f"Basic Type: {bt}"
                )
                all_pairs.append([query_text, bt_str])
                flat_meta.append((i, bt))

        if not all_pairs:
            return [("", 0.0, "zero-shot")] * n

        # 3. Score with cross-encoder (Single Batch Inference)
        raw_logits = self.embed_engine.score_cross_encoder(all_pairs)
        if len(raw_logits) == 0:
            return [("", 0.0, "zero-shot")] * n

        # 4. Calibrate confidence: sigmoid(logit)
        confs = 1.0 / (1.0 + np.exp(-np.array(raw_logits)))

        best_conf_per_sku = [0.0] * n
        best_class_per_sku = [""] * n

        for flat_idx, (sku_idx, bt) in enumerate(flat_meta):
            conf = float(confs[flat_idx])
            if conf > best_conf_per_sku[sku_idx]:
                best_conf_per_sku[sku_idx] = conf
                best_class_per_sku[sku_idx] = bt

        results: list[tuple[str, float, str]] = []
        for i in range(n):
            if best_conf_per_sku[i] >= self.confidence_threshold:
                results.append((best_class_per_sku[i], best_conf_per_sku[i], "zero-shot"))
            else:
                results.append(("", 0.0, "zero-shot"))

        return results


class Tier2FewShotClassifier:
    """
    Tier 2: Few-Shot Instance Matching ($1 <= N_c < 15$ examples).
    Method: Cosine-Weighted k-Nearest Neighbors (k-NN) / Metric Search via Qdrant.
    Formula:
        P(class c | x) = sum_{i in top-k, y_i = c} sim(x, x_i) / sum_{i in top-k} sim(x, x_i)
    Generic Keywords (Multi-label ML-kNN):
        W(w | x) = sum_{i in top-k, w in GK_i} sim(x, x_i) / sum_{i in top-k} sim(x, x_i)
    """

    def __init__(
        self,
        registry: TaxonomyLifecycleRegistry,
        vector_store: Any,
        top_k: int = config.FEW_SHOT_TOP_K,
        confidence_threshold: float = config.REVIEW_THRESHOLD,
        gk_weight_threshold: float = config.FEW_SHOT_GK_WEIGHT_THRESHOLD,
    ):
        self.registry = registry
        self.vector_store = vector_store
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold
        self.gk_weight_threshold = gk_weight_threshold

    def evaluate_neighbors(
        self,
        neighbor_hits: list[dict[str, Any]],
    ) -> tuple[str, float, list[str], str]:
        """
        Computes cosine-weighted class probabilities and multi-label generic keywords from top-k neighbors.
        Returns: (best_few_shot_bt, probability, propagated_gks, source).
        """
        if not neighbor_hits:
            return "", 0.0, [], "few-shot"

        # Filter and extract cosine similarity
        sims = []
        bts = []
        gks_per_hit = []

        for hit in neighbor_hits:
            score = float(hit.get("_qdrant_score_", 0.0))
            clipped_sim = max(0.0, score)
            sims.append(clipped_sim)

            bt = str(hit.get("basictype") or hit.get("BasicType") or "").strip()
            bts.append(bt)

            raw_gk = str(
                hit.get("Generic keywords")
                or hit.get("GenericKeywords")
                or hit.get("generic_keywords")
                or ""
            )
            gks = [tag.strip() for tag in raw_gk.split(",") if tag.strip()]
            gks_per_hit.append(gks)

        total_sim = sum(sims)
        if total_sim <= 1e-9:
            return "", 0.0, [], "few-shot"

        # Calculate class probabilities: P(c | x) = sum_{y_i = c} sim / sum sim
        class_sim_sums: dict[str, float] = {}
        for bt, sim in zip(bts, sims):
            if bt:
                class_sim_sums[bt] = class_sim_sums.get(bt, 0.0) + sim

        # Prioritize classes that reside in Tier 2 (1 <= N_c < 15)
        few_shot_class_probs: dict[str, float] = {}
        all_class_probs: dict[str, float] = {}

        for bt, sim_sum in class_sim_sums.items():
            prob = sim_sum / total_sim
            all_class_probs[bt] = prob
            if self.registry.get_tier(bt) == ClassLifecycleTier.TIER_2_FEW_SHOT:
                few_shot_class_probs[bt] = prob

        # Best candidate
        if few_shot_class_probs:
            best_bt = max(few_shot_class_probs.items(), key=lambda x: x[1])[0]
            best_prob = few_shot_class_probs[best_bt]
        elif all_class_probs:
            best_bt = max(all_class_probs.items(), key=lambda x: x[1])[0]
            best_prob = all_class_probs[best_bt]
        else:
            best_bt = ""
            best_prob = 0.0

        # Multi-label Generic Keyword Propagation (ML-kNN):
        # W(w | x) = sum_{w in GK_i} sim / total_sim
        gk_weights: dict[str, float] = {}
        for gks, sim in zip(gks_per_hit, sims):
            for kw in gks:
                gk_weights[kw] = gk_weights.get(kw, 0.0) + sim

        propagated_gks = [
            kw
            for kw, weight_sum in sorted(gk_weights.items(), key=lambda x: -x[1])
            if (weight_sum / total_sim) >= self.gk_weight_threshold
        ]

        return best_bt, best_prob, propagated_gks, "few-shot"

    def predict(self, dense_vec: np.ndarray, domain: str) -> tuple[str, float, list[str], str]:
        """Queries Qdrant for nearest neighbors and executes Tier 2 few-shot matching."""
        if self.vector_store is None:
            return "", 0.0, [], "few-shot"

        try:
            hits = self.vector_store.search_catalog_neighbors(
                dense_vec, domain=domain, top_k=self.top_k
            )
            return self.evaluate_neighbors(hits)
        except Exception as e:
            logger.warning(f"[{domain}] Tier 2 nearest neighbors query failed: {e}")
            return "", 0.0, [], "few-shot"

    def batch_predict(
        self,
        dense_vecs: list[np.ndarray],
        domain: str,
    ) -> list[tuple[str, float, list[str], str]]:
        """Batch-queries Qdrant and evaluates Tier 2 few-shot matching."""
        n = len(dense_vecs)
        if n == 0:
            return []

        if self.vector_store is None:
            return [("", 0.0, [], "few-shot")] * n

        try:
            batch_hits = self.vector_store.search_batch_catalog_neighbors(
                dense_vecs, domain=domain, top_k=self.top_k
            )
            return [self.evaluate_neighbors(hits) for hits in batch_hits]
        except Exception as e:
            logger.warning(f"[{domain}] Tier 2 batch nearest neighbors query failed: {e}")
            return [("", 0.0, [], "few-shot")] * n


class Tier3CentroidClassifier:
    """
    Tier 3: Warm/Dense Centroids ($N_c >= 15$ examples).
    Method: Class Prototype / Centroid Classifier.
    Predict via temperature-scaled batch matrix multiplication:
        softmax((sku_embedding @ prototypes.T) / tau)
    """

    def __init__(
        self,
        registry: TaxonomyLifecycleRegistry,
        tau: float = config.COLD_START_TAU,
        confidence_threshold: float = config.REVIEW_THRESHOLD,
    ):
        self.registry = registry
        self.tau = tau
        self.confidence_threshold = confidence_threshold

    def predict(self, dense_vec: np.ndarray) -> tuple[str, float, str]:
        """Calculates temperature-scaled class probability against warm centroids."""
        warm_protos = self.registry.warm_centroids
        if warm_protos.shape[0] == 0:
            return "", 0.0, "trained"

        probas, best_indices, confs = EmbeddingEngine.predict_centroids_temperature(
            dense_vec, warm_protos, tau=self.tau
        )
        best_idx = int(best_indices[0])
        best_conf = float(confs[0])
        best_class = self.registry.warm_index_to_class[best_idx]

        return best_class, best_conf, "trained"

    def batch_predict(self, dense_vecs: np.ndarray) -> list[tuple[str, float, str]]:
        """Batch temperature-scaled matrix multiplication against warm centroids."""
        warm_protos = self.registry.warm_centroids
        n = dense_vecs.shape[0]
        if n == 0:
            return []

        if warm_protos.shape[0] == 0:
            return [("", 0.0, "trained")] * n

        probas, best_indices, confs = EmbeddingEngine.predict_centroids_temperature(
            dense_vecs, warm_protos, tau=self.tau
        )

        results = []
        for i in range(n):
            idx = int(best_indices[i])
            conf = float(confs[i])
            cls_name = self.registry.warm_index_to_class[idx]
            results.append((cls_name, conf, "trained"))

        return results


class ColdStartRouter:
    """
    Multi-Tier Cold-Start to Warm-Start Classifier & Fallback Engine.
    Seamlessly routes incoming SKUs across:
      Tier 1: Pure Zero-Shot (N_c = 0)
      Tier 2: Few-Shot Instance Matching (1 <= N_c < 15) + ML-kNN Keyword Propagation
      Tier 3: Warm Centroid Matrix Multiplication (N_c >= 15)
    Acts as a high-precision fallback for ZeroShotClassifier when primary models lack coverage.
    """

    def __init__(
        self,
        domain: str,
        embed_engine: EmbeddingEngine,
        vector_store: Any,
        descriptions: dict[str, Any],
        cat_df: pd.DataFrame | None = None,
        cache_dir: str | None = None,
        few_shot_threshold: int = config.LIFECYCLE_FEW_SHOT_THRESHOLD,
        tau: float = config.COLD_START_TAU,
    ):
        self.domain = domain
        self.embed_engine = embed_engine
        self.vector_store = vector_store
        self.cache_dir = cache_dir or config.CACHE_DIR
        self.tau = tau

        # 1. Initialize Taxonomy Registry
        self.registry = TaxonomyLifecycleRegistry(
            domain=domain,
            cache_dir=self.cache_dir,
            few_shot_threshold=few_shot_threshold,
        )

        bt_descs = descriptions.get("bt_descriptions", {})
        registered_bts = list(bt_descs.keys())
        self.registry.sync_from_catalog(
            registered_bts=registered_bts,
            descriptions=bt_descs,
            cat_df=cat_df,
            embed_engine=embed_engine,
        )

        # 2. Instantiate Tiers
        self.tier1_zero_shot = Tier1ZeroShotClassifier(
            embed_engine=embed_engine,
            registry=self.registry,
            confidence_threshold=config.BT_ZERO_SHOT_CONFIDENCE_THRESHOLD,
        )
        self.tier2_few_shot = Tier2FewShotClassifier(
            registry=self.registry,
            vector_store=vector_store,
            top_k=15,
            confidence_threshold=config.REVIEW_THRESHOLD,
            gk_weight_threshold=0.35,
        )
        self.tier3_centroid = Tier3CentroidClassifier(
            registry=self.registry,
            tau=self.tau,
            confidence_threshold=config.REVIEW_THRESHOLD,
        )

    def register_new_tag(self, tag: str, description: str = ""):
        """Dynamically registers a newly introduced tag (with N_c = 0) at runtime."""
        self.registry.register_tag(
            tag=tag,
            description=description,
            sample_count=0,
            embed_engine=self.embed_engine,
        )
        logger.info(f"[{self.domain.upper()}] Dynamically registered new Zero-Shot tag: '{tag}'")

    def route_single(
        self,
        dense_vec: np.ndarray,
        sku_name: str = "",
        sku_description: str = "",
        price: float | None = None,
    ) -> tuple[str, float, str, list[str]]:
        """
        Routes a single SKU through the multi-tier hierarchy.
        Returns: (predicted_bt, confidence, source, propagated_gks)
        """
        # Step 1: Check Warm Centroids (Tier 3)
        if self.registry.warm_centroids.shape[0] > 0:
            c_tag, c_conf, c_source = self.tier3_centroid.predict(dense_vec)
            # High-confidence in-distribution warm centroid match
            if c_conf >= config.AUTO_THRESHOLD:
                return c_tag, c_conf, c_source, []

        # Step 2: Evaluate Few-Shot Instances in Qdrant (Tier 2)
        few_shot_tag, few_shot_conf, few_shot_gks, few_shot_src = self.tier2_few_shot.predict(
            dense_vec, domain=self.domain
        )
        if (
            few_shot_tag
            and self.registry.get_tier(few_shot_tag) == ClassLifecycleTier.TIER_2_FEW_SHOT
        ):
            if few_shot_conf >= config.REVIEW_THRESHOLD:
                return few_shot_tag, few_shot_conf, few_shot_src, few_shot_gks

        # Step 3: Pure Zero-Shot Cross-Encoder (Tier 1)
        if sku_name and self.registry.zero_shot_classes:
            zs_tag, zs_conf, zs_source = self.tier1_zero_shot.predict(
                sku_name, sku_description, query_dense=dense_vec
            )
            if zs_conf >= config.BT_ZERO_SHOT_CONFIDENCE_THRESHOLD and zs_tag:
                return zs_tag, zs_conf, zs_source, []

        # Step 4: Fallback to best available centroid or few-shot candidate
        if self.registry.warm_centroids.shape[0] > 0:
            c_tag, c_conf, c_source = self.tier3_centroid.predict(dense_vec)
            if c_conf >= config.REVIEW_THRESHOLD:
                return c_tag, c_conf, c_source, []

        if few_shot_tag and few_shot_conf >= config.REVIEW_THRESHOLD:
            return few_shot_tag, few_shot_conf, few_shot_src, few_shot_gks

        return "", 0.0, "zero-shot", []

    def route_batch(
        self,
        dense_vecs: np.ndarray,
        sku_names: list[str] | None = None,
        sku_descriptions: list[str] | None = None,
        prices: list[float] | None = None,
    ) -> list[tuple[str, float, str, list[str]]]:
        """
        Batch version of multi-tier routing.
        Optimized to process warm centroids via BLAS, and selectively invoke Qdrant/Cross-Encoder.
        """
        n = dense_vecs.shape[0]
        if n == 0:
            return []

        if sku_names is None:
            sku_names = [""] * n
        if sku_descriptions is None:
            sku_descriptions = [""] * n

        results: list[tuple[str, float, str, list[str]] | None] = [None] * n

        # Pass 1: Warm Centroid Batch Screening (Tier 3)
        warm_preds = []
        if self.registry.warm_centroids.shape[0] > 0:
            warm_preds = self.tier3_centroid.batch_predict(dense_vecs)

        unresolved_indices = []
        for i in range(n):
            if warm_preds and warm_preds[i][1] >= config.AUTO_THRESHOLD:
                c_tag, c_conf, c_src = warm_preds[i]
                results[i] = (c_tag, c_conf, c_src, [])
            else:
                unresolved_indices.append(i)

        if not unresolved_indices:
            return [r for r in results if r is not None]

        # Pass 2: Few-Shot Instance Retrieval for unresolved items (Tier 2)
        unresolved_vecs = [dense_vecs[i] for i in unresolved_indices]
        few_shot_results = self.tier2_few_shot.batch_predict(unresolved_vecs, domain=self.domain)

        still_unresolved = []
        for list_idx, orig_idx in enumerate(unresolved_indices):
            fs_tag, fs_conf, fs_gks, fs_src = few_shot_results[list_idx]
            if (
                fs_tag
                and self.registry.get_tier(fs_tag) == ClassLifecycleTier.TIER_2_FEW_SHOT
                and fs_conf >= config.REVIEW_THRESHOLD
            ):
                results[orig_idx] = (fs_tag, fs_conf, fs_src, fs_gks)
            else:
                still_unresolved.append((orig_idx, fs_tag, fs_conf, fs_gks, fs_src))

        # Pass 3: Zero-Shot Cross-Encoder for remaining items (Tier 1)
        if still_unresolved and self.registry.zero_shot_classes:
            zs_names = [sku_names[idx] for idx, _, _, _, _ in still_unresolved]
            zs_descs = [sku_descriptions[idx] for idx, _, _, _, _ in still_unresolved]
            zs_vecs = np.vstack([dense_vecs[idx] for idx, _, _, _, _ in still_unresolved])

            zs_preds = self.tier1_zero_shot.batch_predict(
                zs_names, zs_descs, query_dense_matrix=zs_vecs
            )
        else:
            zs_preds = [("", 0.0, "zero-shot")] * len(still_unresolved)

        for list_idx, (orig_idx, fs_tag, fs_conf, fs_gks, fs_src) in enumerate(still_unresolved):
            zs_tag, zs_conf, zs_src = zs_preds[list_idx]

            # Try Tier 1 Cross-Encoder
            if zs_conf >= config.BT_ZERO_SHOT_CONFIDENCE_THRESHOLD and zs_tag:
                results[orig_idx] = (zs_tag, zs_conf, zs_src, [])
            # Fallback to moderate warm centroid
            elif warm_preds and warm_preds[orig_idx][1] >= config.REVIEW_THRESHOLD:
                c_tag, c_conf, c_src = warm_preds[orig_idx]
                results[orig_idx] = (c_tag, c_conf, c_src, [])
            # Fallback to moderate few-shot
            elif fs_tag and fs_conf >= config.REVIEW_THRESHOLD:
                results[orig_idx] = (fs_tag, fs_conf, fs_src, fs_gks)
            # Desperate fallback to low-confidence warm centroid
            elif warm_preds and warm_preds[orig_idx][0]:
                c_tag, c_conf, c_src = warm_preds[orig_idx]
                results[orig_idx] = (c_tag, c_conf, "zero-shot", [])
            else:
                results[orig_idx] = ("", 0.0, "zero-shot", [])

        return [r for r in results if r is not None]
