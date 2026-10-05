import hashlib
import json
import logging
import os

import joblib
import numpy as np
import pandas as pd

from engine import config
from engine.data_pipeline.cache_manager import calculate_df_hash
from engine.utils.flavor_utils import build_food_flavors_info

logger = logging.getLogger("matchops.classifier")


class ZeroShotClassifier:
    """
    Classifies SKU Basic Type and a third tag (Region/Category) using bge-m3 embeddings.

    Modes (auto-selected at startup):
      • trained  — LogisticRegression on bge-m3 embeddings, trained from
                   data/training_data.csv.  Most accurate.
      • zero-shot— Cosine similarity to BT/Third Tag description embeddings.
                   Works with zero labeled data.

    Third Tag overrides always run first to handle
    dataset-specific quirks that the model cannot know from food names alone.
    """

    def __init__(
        self,
        model,
        domain: str,
        descriptions: dict,
        cache_dir: str | None = None,
        cat_df: pd.DataFrame = None,
        brands_df: pd.DataFrame = None,
        force_retrain: bool = False,
    ):
        self.model = model
        self.domain = domain
        self.cache_dir = cache_dir or config.CACHE_DIR
        self._trained = False
        self._price_scaler = None
        self.cat_df = cat_df if cat_df is not None else pd.DataFrame()
        self.brands_df = brands_df if brands_df is not None else pd.DataFrame()
        self.food_flavors_dict, _, _, _, _ = build_food_flavors_info(brands_df)

        # Always use this for SKU query embedding so training and inference are consistent.
        self._active_model = model

        self.bt_descs = descriptions.get("bt_descriptions", {})
        self.third_tag_descs = descriptions.get("third_tag_descriptions", {})
        self.third_tag_overrides = descriptions.get("third_tag_overrides", {})
        self.bt_to_gk_umbrella = descriptions.get("bt_to_gk_umbrella", {})
        self.bt_gk_map = descriptions.get("bt_gk_map", {})

        self.bt_labels = list(self.bt_descs.keys())
        self.bt_embs_pure = self._embed_cached(
            list(self.bt_descs.keys()), f"{domain}_classifier_bt_pure"
        )
        self.bt_embs_desc = self._embed_cached(
            list(self.bt_descs.values()), f"{domain}_classifier_bt_descs"
        )

        self.third_tag_labels = list(self.third_tag_descs.keys())
        self.third_tag_embs_pure = self._embed_cached(
            list(self.third_tag_descs.keys()), f"{domain}_classifier_third_tag_pure"
        )
        self.third_tag_embs_desc = self._embed_cached(
            list(self.third_tag_descs.values()), f"{domain}_classifier_third_tag_descs"
        )

        # BasicType (BT) Model Configuration: "arcface" or "logreg"
        self.bt_model = config.get_bt_model(domain)
        self._arcface_session = None
        self._arcface_classes = []
        self._arcface_price_scaler = None
        self._arcface_input_name = None

        if self.bt_model == "arcface":
            self._load_arcface_model()

        # Attempt to train sklearn classifiers if labeled data exists
        self._try_train(force_retrain=force_retrain)

        # Multi-Tier Cold-Start to Warm-Start Router (Tier 1 Zero-Shot, Tier 2 Few-Shot, Tier 3 Centroids)
        self.cold_start_router = None
        try:
            from engine.classification.cold_start_router import ColdStartRouter
            from engine.data_pipeline.vector_store import VectorStore

            vs = None
            try:
                vs = VectorStore()
            except Exception as vs_err:
                logger.debug(
                    f"[{domain}] VectorStore connection notice for ColdStartRouter: {vs_err}"
                )

            self.cold_start_router = ColdStartRouter(
                domain=domain,
                embed_engine=self.model,
                vector_store=vs,
                descriptions=descriptions,
                cat_df=self.cat_df,
                cache_dir=self.cache_dir,
            )
            logger.info(
                f"[{domain.upper()}] ColdStartRouter initialized successfully as classifier fallback."
            )
        except Exception as router_err:
            logger.warning(
                f"[{domain.upper()}] ColdStartRouter initialization failed: {router_err}"
            )

    def _load_arcface_model(self):
        """
        Loads the INT8 ArcFace ONNX model and label mapping for the domain.
        Fails fast if the model file or label mapping is missing or corrupt.
        """
        import onnxruntime as ort

        onnx_path = config.get_bt_arcface_onnx_path(self.domain)
        labels_path = config.get_bt_arcface_labels_path(self.domain)

        if not os.path.exists(onnx_path) or not os.path.exists(labels_path):
            raise RuntimeError(
                f"[FAIL-FAST] Domain '{self.domain}' is configured with BT_MODEL='arcface', "
                f"but required ONNX model file '{onnx_path}' or label mapping '{labels_path}' is missing. "
                f"Please train and export the model using 'python -m scripts.ml.train_bt_head --domain {self.domain}' "
                f"or configure {self.domain.upper()}_BT_MODEL=logreg in your environment."
            )

        try:
            with open(labels_path, encoding="utf-8") as f:
                meta = json.load(f)
            self._arcface_classes = meta["classes"]
            scaler_info = meta.get("price_scaler", {})
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            scaler.mean_ = np.array([scaler_info.get("mean", 0.0)], dtype=np.float32)
            scaler.scale_ = np.array([scaler_info.get("scale", 1.0)], dtype=np.float32)
            scaler.var_ = np.array([scaler_info.get("var", 1.0)], dtype=np.float32)
            self._arcface_price_scaler = scaler
        except Exception as e:
            raise RuntimeError(
                f"[FAIL-FAST] Failed to parse ArcFace label metadata from '{labels_path}': {e}"
            ) from e

        try:
            sess_opts = ort.SessionOptions()
            sess_opts.add_session_config_entry("session.use_mmap_for_weights", "1")
            self._arcface_session = ort.InferenceSession(
                onnx_path, sess_options=sess_opts, providers=["CPUExecutionProvider"]
            )
            self._arcface_input_name = self._arcface_session.get_inputs()[0].name
            logger.info(
                f"[ARCFACE] [{self.domain.upper()}] Successfully loaded ArcFace INT8 model "
                f"from {onnx_path} with {len(self._arcface_classes)} classes."
            )
        except Exception as e:
            raise RuntimeError(
                f"[FAIL-FAST] Failed to initialize ONNX InferenceSession from '{onnx_path}': {e}"
            ) from e

    def _preprocess_arcface_prices(self, prices_list: list[float]) -> np.ndarray:
        """Preprocesses prices using the ArcFace model's fitted StandardScaler."""
        prices = np.array(prices_list, dtype=np.float32).reshape(-1, 1)
        prices = np.clip(prices, 0.0, None)
        log_prices = np.log1p(prices)
        if self._arcface_price_scaler is not None:
            scaled = self._arcface_price_scaler.transform(log_prices)
            zero_mask = (prices <= 0.0).flatten()
            scaled[zero_mask] = 0.0
            return scaled.astype(np.float32)
        return np.zeros_like(log_prices, dtype=np.float32)

    # ── Internal helpers ──────────────────────────────────────

    def _embed_cached(self, texts: list[str], cache_key: str) -> np.ndarray:
        """Embed a list of texts with disk caching."""
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        fingerprint = hashlib.md5("|".join(texts).encode()).hexdigest()

        if os.path.exists(cache_file):
            try:
                cached = joblib.load(cache_file)
                if cached.get("fingerprint") == fingerprint:
                    return cached["embeddings"]
            except Exception:
                pass

        out = self.model.encode(texts)
        embs = out["dense"]
        joblib.dump({"fingerprint": fingerprint, "embeddings": embs}, cache_file)
        return embs

    def _embed_weighted_sku_incremental(
        self,
        names_list: list[str],
        descs_list: list[str],
        cats_list: list[str],
    ) -> np.ndarray:
        """
        Embed training SKUs incrementally, caching the computed dense vectors.
        This bypasses the heavy embedding model evaluation for unchanged SKUs.
        """
        from engine.config import CLASSIFIER_WEIGHTS

        if not names_list:
            return np.empty((0, 1024))

        cache_file = os.path.join(self.cache_dir, f"{self.domain}_weighted_skus_cache.pkl")
        os.makedirs(self.cache_dir, exist_ok=True)

        sku_cache = {}
        if os.path.exists(cache_file):
            try:
                sku_cache = joblib.load(cache_file)
            except Exception:
                sku_cache = {}

        # Construct lookup keys for current training set
        keys = [
            f"{n.strip()}||{d.strip()}||{c.strip()}"
            for n, d, c in zip(names_list, descs_list, cats_list)
        ]

        # Find which items are missing from cache
        missing_indices = [i for i, key in enumerate(keys) if key not in sku_cache]

        if missing_indices:
            logger.info(
                f"[EMBED] {self.domain} classifier training: {len(missing_indices)} items to encode "
                f"({len(keys) - len(missing_indices)} cached)."
            )
            # Gather missing inputs
            missing_names = [names_list[i] for i in missing_indices]
            missing_descs = [descs_list[i] for i in missing_indices]
            missing_cats = [cats_list[i] for i in missing_indices]

            # Batch encode the missing ones
            embs = self.model.embed_weighted_sku(
                missing_names, missing_descs, missing_cats, weights=CLASSIFIER_WEIGHTS
            )
            dense_vectors = embs["dense"]

            # Store in cache
            for idx, i in enumerate(missing_indices):
                key = keys[i]
                sku_cache[key] = dense_vectors[idx]

            # Save the updated cache
            joblib.dump(sku_cache, cache_file)
            logger.info(f"[EMBED] {self.domain} classifier training: cache updated.")
        else:
            logger.info(
                f"[EMBED] {self.domain} classifier training: all {len(keys)} training embeddings served from cache."
            )

        # Reassemble the training matrix X
        X = np.vstack([sku_cache[key] for key in keys])
        return X

    def _preprocess_prices(self, prices_list: list[float], is_training: bool = False) -> np.ndarray:
        """Log-scales prices and applies/fits StandardScaler."""
        prices = np.array(prices_list, dtype=np.float32).reshape(-1, 1)
        prices = np.clip(prices, 0.0, None)
        log_prices = np.log1p(prices)

        if is_training:
            from sklearn.preprocessing import StandardScaler

            self._price_scaler = StandardScaler()
            scaled_prices = self._price_scaler.fit_transform(log_prices)
        else:
            if hasattr(self, "_price_scaler") and self._price_scaler is not None:
                scaled_prices = self._price_scaler.transform(log_prices)
                # When price is 0.0 or not provided during inference, log1p(0) minus mean
                # produces an extreme outlier z-score (~ -7.5) that corrupts text-based classification.
                # Neutralize missing/zero prices by setting scaled price to 0.0 (the dataset mean).
                zero_mask = (prices <= 0.0).flatten()
                scaled_prices[zero_mask] = 0.0
            else:
                scaled_prices = np.zeros_like(log_prices)
        return scaled_prices

    def _try_train(self, force_retrain: bool = False):
        if self.cat_df.empty:
            logger.warning("[TRAIN] Empty catalog dataframe — using zero-shot mode")
            return

        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.multiclass import OneVsRestClassifier
            from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

            cache_file = os.path.join(self.cache_dir, f"{self.domain}_classifier_model.pkl")
            HASH_STORAGE_PATH = os.path.join(config.CACHE_DIR, "model_hashes.json")
            domain_hash_key = f"{self.domain}_training_state"

            current_hash = calculate_df_hash(self.cat_df, domain=self.domain)
            old_bt_clf, old_bt_enc = None, None
            old_third_tag_clf, old_third_tag_enc = None, None

            # Load from cache if cache exists and not forced
            if os.path.exists(cache_file):
                try:
                    cached = joblib.load(cache_file)
                    if not force_retrain:
                        self._bt_enc = cached["bt_enc"]
                        self._bt_clf = cached["bt_clf"]
                        self._third_tag_enc = cached["third_tag_enc"]
                        self._third_tag_clf = cached["third_tag_clf"]
                        self._gk_enc = cached.get("gk_enc")
                        self._gk_clf = cached.get("gk_clf")
                        self._price_scaler = cached.get("price_scaler")
                        self._trained = True
                        logger.info(f"[TRAIN] Loaded classifier model from cache ({self.domain}).")
                        return
                    else:
                        old_bt_enc = cached.get("bt_enc")
                        old_bt_clf = cached.get("bt_clf")
                        old_third_tag_enc = cached.get("third_tag_enc")
                        old_third_tag_clf = cached.get("third_tag_clf")
                except Exception as e:
                    logger.warning(f"[TRAIN] Cache corrupt for {self.domain}, retraining: {e}")

            logger.info(f"[TRAIN] Training classifier model for {self.domain}...")

            df = self.cat_df.fillna("")

            # Drop rows with no BT or Third Tag label
            from engine.config import (
                COL_DESCRIPTION,
                COL_GK,
                COL_INPUT_CATEGORY,
                COL_NAME,
                get_third_tag_col,
            )

            target_col = get_third_tag_col(self.domain)
            missing = {"Name", "basictype", target_col, COL_GK} - set(df.columns)
            if missing:
                logger.warning(f"[TRAIN] ⚠ Catalog missing columns: {missing} — using zero-shot")
                return

            df = df[df["basictype"].str.strip() != ""]
            df = df[df[target_col].str.strip() != ""]
            df = df[df[COL_GK].str.strip() != ""]

            if len(df) < 10:
                logger.warning(
                    f"[TRAIN] ⚠ Only {len(df)} labeled rows — need ≥10. Using zero-shot."
                )
                return

            # Build query strings exactly matching inference (weighted multi-field embedding)
            names_list = df[COL_NAME].astype(str).str.strip().tolist()
            descs_list = (
                df[COL_DESCRIPTION].astype(str).str.strip().tolist()
                if COL_DESCRIPTION in df.columns
                else [""] * len(df)
            )
            col_cat = (
                COL_INPUT_CATEGORY
                if COL_INPUT_CATEGORY in df.columns
                else ("category" if "category" in df.columns else "")
            )
            cats_list = df[col_cat].astype(str).str.strip().tolist() if col_cat else [""] * len(df)

            logger.info(f"[TRAIN] Training on {len(df)} labeled SKUs...")

            # Retrieve embeddings using the incremental disk cache helper
            X = self._embed_weighted_sku_incremental(names_list, descs_list, cats_list)

            # Preprocess and append price feature to X
            prices_list = pd.to_numeric(df["Price"], errors="coerce").fillna(0.0).tolist()
            scaled_prices = self._preprocess_prices(prices_list, is_training=True)
            X = np.hstack([X, scaled_prices])

            # ── BT classifier ──────────────────────────────────
            # NOTE: class_weight="balanced" is intentionally NOT used here.
            # With ~1530 BT classes and only ~6 samples/class, lbfgs silently
            # converges to all-zero weights under balanced weighting, causing
            # every SKU to predict class index 0 (the first alphabetically).
            bt_raw = df["basictype"].str.strip().tolist()
            self._bt_enc = LabelEncoder().fit(bt_raw)
            y_bt = self._bt_enc.transform(bt_raw)

            # Check if we can warm start BT classifier (LogisticRegression)
            can_warm_start_bt = False
            if (
                old_bt_clf is not None
                and old_bt_enc is not None
                and hasattr(old_bt_clf, "coef_")
                and list(old_bt_enc.classes_) == list(self._bt_enc.classes_)
            ):
                can_warm_start_bt = True

            if can_warm_start_bt:
                self._bt_clf = old_bt_clf
                self._bt_clf.warm_start = True
                self._bt_clf.fit(X, y_bt)
            else:
                self._bt_clf = LogisticRegression(max_iter=1000, C=5.0).fit(X, y_bt)

            # ── Third Tag classifier ──────────────────────────────
            third_tag_raw = df[target_col].str.strip().tolist()
            self._third_tag_enc = LabelEncoder().fit(third_tag_raw)
            y_third_tag = self._third_tag_enc.transform(third_tag_raw)

            # Check if we can warm start Third Tag classifier (LogisticRegression)
            can_warm_start_third = False
            if (
                old_third_tag_clf is not None
                and old_third_tag_enc is not None
                and hasattr(old_third_tag_clf, "coef_")
                and list(old_third_tag_enc.classes_) == list(self._third_tag_enc.classes_)
            ):
                can_warm_start_third = True

            if can_warm_start_third:
                self._third_tag_clf = old_third_tag_clf
                self._third_tag_clf.warm_start = True
                self._third_tag_clf.fit(X, y_third_tag)
            else:
                self._third_tag_clf = LogisticRegression(
                    max_iter=1000, C=5.0, class_weight="balanced"
                ).fit(X, y_third_tag)

            # ── GK classifier (Multi-label) ───────────────────────
            gk_raw = [
                [tag.strip() for tag in tags.split(",") if tag.strip()] for tags in df[COL_GK]
            ]
            self._gk_enc = MultiLabelBinarizer()
            y_gk = self._gk_enc.fit_transform(gk_raw)
            base_clf = LogisticRegression(max_iter=250, C=5.0, class_weight="balanced")
            self._gk_clf = OneVsRestClassifier(base_clf).fit(X, y_gk)

            n_bt = len(set(bt_raw))
            n_third_tag = len(set(third_tag_raw))
            n_gk_tags = len(self._gk_enc.classes_)
            self._trained = True
            logger.info(
                f"[TRAIN] ✓ Trained — "
                f"{len(df)} examples | {n_bt} BTs | {n_third_tag} {target_col.capitalize()}s | {n_gk_tags} GK tags"
            )

            # Save to cache
            os.makedirs(self.cache_dir, exist_ok=True)
            joblib.dump(
                {
                    "bt_enc": self._bt_enc,
                    "bt_clf": self._bt_clf,
                    "third_tag_enc": self._third_tag_enc,
                    "third_tag_clf": self._third_tag_clf,
                    "gk_enc": self._gk_enc,
                    "gk_clf": self._gk_clf,
                    "price_scaler": self._price_scaler,
                },
                cache_file,
            )

            if current_hash:
                try:
                    stored_hashes = {}
                    if os.path.exists(HASH_STORAGE_PATH):
                        try:
                            with open(HASH_STORAGE_PATH, encoding="utf-8") as f:
                                stored_hashes = json.load(f)
                        except Exception:
                            stored_hashes = {}
                    stored_hashes[domain_hash_key] = current_hash
                    with open(HASH_STORAGE_PATH, "w", encoding="utf-8") as f:
                        json.dump(stored_hashes, f, indent=4)
                except Exception as hash_err:
                    logger.warning(f"[TRAIN] Failed to write model hash: {hash_err}")

        except Exception as e:
            logger.error(
                f"[TRAIN] ⚠ Classifier training failed: {e} — using zero-shot", exc_info=True
            )

    # ── Public API ────────────────────────────────────────────

    @property
    def active_bt_model(self) -> str:
        """Returns the active BasicType classification model identifier ('arcface' or 'logreg')."""
        return getattr(self, "bt_model", "logreg")

    def register_new_tag(self, tag: str, description: str = ""):
        """Dynamically registers a newly introduced tag (with N_c = 0) into the classifier and router."""
        clean_tag = tag.strip()
        if not clean_tag:
            return
        if clean_tag not in self.bt_labels:
            self.bt_labels.append(clean_tag)
        self.bt_descs[clean_tag] = description or clean_tag
        if hasattr(self, "cold_start_router") and self.cold_start_router is not None:
            self.cold_start_router.register_new_tag(clean_tag, description)

    def predict_bt(
        self,
        vec: np.ndarray,
        price: float | None = None,
        sku_name: str = "",
        sku_description: str = "",
    ) -> tuple[str, float, str, list[str]]:
        """
        Returns (bt_label, confidence, source, propagated_gks).
        source is one of: 'trained', 'few-shot', 'zero-shot'
        """
        vec_2d = vec.reshape(1, -1) if vec.ndim == 1 else vec
        p_val = float(price) if price is not None else 0.0

        # Check for dynamic zero-shot classes that may match with high cross-encoder confidence
        cold_router = getattr(self, "cold_start_router", None)
        if cold_router is not None and cold_router.registry.zero_shot_classes and sku_name:
            zs_tag, zs_conf, zs_src = cold_router.tier1_zero_shot.predict(
                sku_name, sku_description, query_dense=vec_2d[0]
            )
            if zs_conf >= config.AUTO_THRESHOLD and zs_tag:
                return zs_tag, zs_conf, zs_src, []

        # 1. Deep Metric Learning ArcFace Model Path
        if self.bt_model == "arcface" and getattr(self, "_arcface_session", None) is not None:
            scaled_p = self._preprocess_arcface_prices([p_val])
            vec_with_price = np.hstack([vec_2d, scaled_p]).astype(np.float32)
            logits = self._arcface_session.run(None, {self._arcface_input_name: vec_with_price})[0]
            # Softmax to derive normalized probabilities
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probas = (exp_logits / np.sum(exp_logits, axis=-1, keepdims=True))[0]
            best = int(np.argmax(probas))
            conf = float(probas[best])
            if conf >= 0.4:
                return self._arcface_classes[best], conf, "trained", []

        # 2. Logistic Regression Model Path
        elif (
            self.bt_model == "logreg"
            and getattr(self, "_trained", False)
            and getattr(self, "_bt_clf", None) is not None
        ):
            scaled_p = self._preprocess_prices([p_val], is_training=False)
            vec_with_price = np.hstack([vec_2d, scaled_p])
            proba = self._bt_clf.predict_proba(vec_with_price)[0]
            best = int(np.argmax(proba))
            conf = float(proba[best])
            if conf >= 0.4:
                return self._bt_enc.classes_[best], conf, "trained", []

        # 3. Cold-Start to Warm-Start Multi-Tier Fallback (Tier 1 Zero-Shot, Tier 2 Few-Shot, Tier 3 Centroids)
        if cold_router is not None:
            r_tag, r_conf, r_src, r_gks = cold_router.route_single(
                vec_2d[0], sku_name=sku_name, sku_description=sku_description, price=price
            )
            if r_tag and r_conf >= config.BT_ZERO_SHOT_CONFIDENCE_THRESHOLD:
                return r_tag, r_conf, r_src, r_gks

        # 4. Zero-shot static fallback: cosine similarity to BT description embeddings
        bt_labels = getattr(self, "bt_labels", [])
        if not bt_labels or not hasattr(self, "bt_embs_pure") or not hasattr(self, "bt_embs_desc"):
            return "", 0.0, "zero-shot", []

        scores_pure = (vec_2d @ self.bt_embs_pure.T)[0]
        scores_desc = (vec_2d @ self.bt_embs_desc.T)[0]
        scores = np.maximum(scores_pure, scores_desc)
        best = int(np.argmax(scores))
        return bt_labels[best], float(scores[best]), "zero-shot", []

    def batch_predict_bt(
        self,
        vecs: np.ndarray,
        prices: list[float],
        sku_names: list[str] | None = None,
        sku_descriptions: list[str] | None = None,
    ) -> list[tuple[str, float, str, list[str]]]:
        """
        Batch version of predict_bt.
        Returns a list of (bt_label, confidence, source, propagated_gks) tuples.
        """
        if vecs.shape[0] == 0:
            return []

        n = len(vecs)
        results = [None] * n

        if sku_names is None:
            sku_names = [""] * n
        if sku_descriptions is None:
            sku_descriptions = [""] * n

        zero_shot_indices = []

        # 1. Deep Metric Learning ArcFace Model Path
        if self.bt_model == "arcface" and getattr(self, "_arcface_session", None) is not None:
            scaled_p = self._preprocess_arcface_prices(prices)
            vecs_with_price = np.hstack([vecs, scaled_p]).astype(np.float32)
            logits = self._arcface_session.run(None, {self._arcface_input_name: vecs_with_price})[0]
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probas = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            bests = np.argmax(probas, axis=1)
            confs = np.max(probas, axis=1)

            for i in range(len(vecs)):
                if confs[i] >= config.BT_TRAINED_CONFIDENCE_THRESHOLD:
                    results[i] = (self._arcface_classes[bests[i]], float(confs[i]), "trained", [])
                else:
                    zero_shot_indices.append(i)

        # 2. Logistic Regression Model Path
        elif (
            self.bt_model == "logreg"
            and getattr(self, "_trained", False)
            and getattr(self, "_bt_clf", None) is not None
        ):
            scaled_p = self._preprocess_prices(prices, is_training=False)
            vec_with_price = np.hstack([vecs, scaled_p])
            probas = self._bt_clf.predict_proba(vec_with_price)
            bests = np.argmax(probas, axis=1)
            confs = np.max(probas, axis=1)

            for i in range(len(vecs)):
                if confs[i] >= config.BT_TRAINED_CONFIDENCE_THRESHOLD:
                    results[i] = (self._bt_enc.classes_[bests[i]], float(confs[i]), "trained", [])
                else:
                    zero_shot_indices.append(i)
        else:
            zero_shot_indices = list(range(len(vecs)))

        # 3. Cold-Start Multi-Tier Fallback for unconfident or cold classes
        cold_router = getattr(self, "cold_start_router", None)
        if zero_shot_indices:
            unresolved_zs = []
            if cold_router is not None:
                zs_vecs = vecs[zero_shot_indices]
                zs_names = [sku_names[i] for i in zero_shot_indices]
                zs_descs = [sku_descriptions[i] for i in zero_shot_indices]
                zs_prices = [prices[i] if i < len(prices) else 0.0 for i in zero_shot_indices]

                router_preds = cold_router.route_batch(
                    zs_vecs, sku_names=zs_names, sku_descriptions=zs_descs, prices=zs_prices
                )
                for idx_in_zs, orig_idx in enumerate(zero_shot_indices):
                    r_tag, r_conf, r_src, r_gks = router_preds[idx_in_zs]
                    if r_tag and r_conf >= config.BT_ZERO_SHOT_CONFIDENCE_THRESHOLD:
                        results[orig_idx] = (r_tag, float(r_conf), r_src, r_gks)
                    else:
                        unresolved_zs.append(orig_idx)
            else:
                unresolved_zs = zero_shot_indices

            # 4. Final static description cosine fallback for any still-unresolved predictions
            if unresolved_zs:
                bt_labels = getattr(self, "bt_labels", [])
                if (
                    not bt_labels
                    or not hasattr(self, "bt_embs_pure")
                    or not hasattr(self, "bt_embs_desc")
                ):
                    for i in unresolved_zs:
                        results[i] = ("", 0.0, "zero-shot", [])
                else:
                    fallback_vecs = vecs[unresolved_zs]
                    scores_pure = fallback_vecs @ self.bt_embs_pure.T
                    scores_desc = fallback_vecs @ self.bt_embs_desc.T
                    scores = np.maximum(scores_pure, scores_desc)
                    bests = np.argmax(scores, axis=1)
                    confs = np.max(scores, axis=1)
                    for idx_in_fb, orig_idx in enumerate(unresolved_zs):
                        results[orig_idx] = (
                            bt_labels[bests[idx_in_fb]],
                            float(confs[idx_in_fb]),
                            "zero-shot",
                            [],
                        )

        return results

    def predict_gk(
        self, vec: np.ndarray, price: float | None = None
    ) -> tuple[list[str], float, str]:
        """
        Returns (list_of_gk_tags, confidence, source).
        Confidence is the average probability of the predicted tags.
        Returns empty list if not trained or if no tags meet the threshold.
        """
        if not self._trained or getattr(self, "_gk_clf", None) is None:
            return [], 0.0, "zero-shot"

        vec_2d = vec.reshape(1, -1) if vec.ndim == 1 else vec
        p_val = float(price) if price is not None else 0.0
        scaled_p = self._preprocess_prices([p_val], is_training=False)
        vec_with_price = np.hstack([vec_2d, scaled_p])

        proba = self._gk_clf.predict_proba(vec_with_price)[0]
        threshold = config.GK_TRAINED_CONFIDENCE_THRESHOLD
        predicted_indices = np.where(proba >= threshold)[0]

        if len(predicted_indices) == 0:
            return [], 0.0, "zero-shot"

        tags = self._gk_enc.classes_[predicted_indices].tolist()
        conf = float(np.mean(proba[predicted_indices]))
        return tags, conf, "trained"

    def batch_predict_gk(
        self, vecs: np.ndarray, prices: list[float]
    ) -> list[tuple[list[str], float, str]]:
        """
        Batch version of predict_gk.
        Returns a list of (list_of_gk_tags, confidence, source) tuples.
        """
        if len(vecs) == 0:
            return []

        if not self._trained or getattr(self, "_gk_clf", None) is None:
            return [([], 0.0, "zero-shot") for _ in range(len(vecs))]

        scaled_p = self._preprocess_prices(prices, is_training=False)
        vecs_with_price = np.hstack([vecs, scaled_p])
        probas = self._gk_clf.predict_proba(vecs_with_price)
        threshold = config.GK_TRAINED_CONFIDENCE_THRESHOLD

        results = []
        for i in range(len(vecs)):
            proba = probas[i]
            predicted_indices = np.where(proba >= threshold)[0]
            if len(predicted_indices) == 0:
                results.append(([], 0.0, "zero-shot"))
            else:
                tags = self._gk_enc.classes_[predicted_indices].tolist()
                conf = float(np.mean(proba[predicted_indices]))
                results.append((tags, conf, "trained"))
        return results

    def predict_third_tag(
        self,
        vec: np.ndarray,
        name: str,
        description: str = "",
        predicted_bt: str = "",
        price: float | None = None,
    ) -> tuple[str, float, str]:
        """
        Returns (third_tag_label, confidence, source).
        source is one of: 'override', 'trained', 'zero-shot'

        Override priority:
          1. BT-keyed override (mined from catalog, covers both Food→Region and Market→Category).
             third_tag_overrides is {bt_label: third_tag_label}.
          2. Trained LogisticRegression (if enough labeled data).
          3. Zero-shot cosine similarity to description embeddings.
        """
        # 1. BT-keyed override (O(1) dict lookup)
        if predicted_bt and predicted_bt in self.third_tag_overrides:
            return self.third_tag_overrides[predicted_bt], 1.0, "override"

        vec_2d = vec.reshape(1, -1) if vec.ndim == 1 else vec

        if self._trained:
            p_val = float(price) if price is not None else 0.0
            scaled_p = self._preprocess_prices([p_val], is_training=False)
            vec_with_price = np.hstack([vec_2d, scaled_p])
            proba = self._third_tag_clf.predict_proba(vec_with_price)[0]
            best = int(np.argmax(proba))
            conf = float(proba[best])
            if conf >= 0.4:
                return self._third_tag_enc.classes_[best], conf, "trained"

        # Zero-shot
        if not self.third_tag_labels:
            return "", 0.0, "zero-shot"

        scores_pure = (vec_2d @ self.third_tag_embs_pure.T)[0]
        scores_desc = (vec_2d @ self.third_tag_embs_desc.T)[0]
        scores = np.maximum(scores_pure, scores_desc)
        best = int(np.argmax(scores))
        return self.third_tag_labels[best], float(scores[best]), "zero-shot"

    def batch_predict_third_tag(
        self,
        vecs: np.ndarray,
        names: list[str] = None,
        descriptions: list[str] = None,
        predicted_bts: list[str] = None,
        prices: list[float] = None,
    ) -> list[tuple[str, float, str]]:
        """
        Batch version of predict_third_tag.
        Returns a list of (third_tag_label, confidence, source) tuples.
        """
        n = len(vecs)
        if n == 0:
            return []

        if predicted_bts is None:
            predicted_bts = [""] * n
        if prices is None:
            prices = [0.0] * n

        results: list[tuple[str, float, str] | None] = [None] * n
        remaining_indices = []

        # 1. BT-keyed override (O(1) dict lookup)
        for i in range(n):
            bt = predicted_bts[i]
            if bt and bt in self.third_tag_overrides:
                results[i] = (self.third_tag_overrides[bt], 1.0, "override")
            else:
                remaining_indices.append(i)

        if not remaining_indices:
            return results

        # 2. Trained LogisticRegression
        zero_shot_indices = []
        if self._trained:
            rem_vecs = vecs[remaining_indices]
            rem_prices = [prices[i] for i in remaining_indices]
            scaled_p = self._preprocess_prices(rem_prices, is_training=False)
            vecs_with_price = np.hstack([rem_vecs, scaled_p])
            probas = self._third_tag_clf.predict_proba(vecs_with_price)
            bests = np.argmax(probas, axis=1)
            confs = np.max(probas, axis=1)

            for list_idx, orig_idx in enumerate(remaining_indices):
                if confs[list_idx] >= 0.4:
                    results[orig_idx] = (
                        self._third_tag_enc.classes_[bests[list_idx]],
                        float(confs[list_idx]),
                        "trained",
                    )
                else:
                    zero_shot_indices.append(orig_idx)
        else:
            zero_shot_indices = remaining_indices

        # 3. Zero-shot cosine similarity
        if zero_shot_indices:
            if not self.third_tag_labels:
                for idx in zero_shot_indices:
                    results[idx] = ("", 0.0, "zero-shot")
            else:
                zs_vecs = vecs[zero_shot_indices]
                scores_pure = zs_vecs @ self.third_tag_embs_pure.T
                scores_desc = zs_vecs @ self.third_tag_embs_desc.T
                scores = np.maximum(scores_pure, scores_desc)
                bests = np.argmax(scores, axis=1)
                confs = np.max(scores, axis=1)
                for list_idx, orig_idx in enumerate(zero_shot_indices):
                    results[orig_idx] = (
                        self.third_tag_labels[bests[list_idx]],
                        float(confs[list_idx]),
                        "zero-shot",
                    )

        return results

    def get_guaranteed_gk(self, bt: str) -> list[str]:
        """
        Return umbrella GK tags for a BT.
        e.g. bt='Iced Coffee' → ['Iced Coffee', 'Beverage', 'Coffee']
        These are tags the semantic GK search cannot reliably find on its own.
        """
        return self.bt_to_gk_umbrella.get(bt, [])
