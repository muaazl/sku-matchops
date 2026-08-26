import hashlib
import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

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

    def __init__(self, model, domain: str, descriptions: dict, cache_dir: str | None = None, cat_df: pd.DataFrame = None, brands_df: pd.DataFrame = None, force_retrain: bool = False):
        self.model     = model
        self.domain    = domain
        self.cache_dir = cache_dir or config.CACHE_DIR
        self._trained  = False
        self._price_scaler = None
        self.cat_df = cat_df if cat_df is not None else pd.DataFrame()
        self.brands_df = brands_df if brands_df is not None else pd.DataFrame()
        self.food_flavors_dict, _, _, _, _ = build_food_flavors_info(brands_df)
        
        # _active_model: may be replaced by SetFit fine-tuned encoder for market domain.
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

        # Attempt to train sklearn classifiers if labeled data exists
        self._try_train(force_retrain=force_retrain)

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
        embs = out['dense']
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
        keys = [f"{n.strip()}||{d.strip()}||{c.strip()}" for n, d, c in zip(names_list, descs_list, cats_list)]

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
            logger.info(f"[EMBED] {self.domain} classifier training: all {len(keys)} training embeddings served from cache.")

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
            else:
                scaled_prices = np.zeros_like(log_prices)
        return scaled_prices

    def _try_train(self, force_retrain: bool = False):
        if self.cat_df.empty:
            logger.warning("[TRAIN] Empty catalog dataframe — using zero-shot mode")
            return

        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
            from sklearn.multiclass import OneVsRestClassifier

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
                        self._bt_enc        = cached["bt_enc"]
                        self._bt_clf        = cached["bt_clf"]
                        self._third_tag_enc = cached["third_tag_enc"]
                        self._third_tag_clf = cached["third_tag_clf"]
                        self._gk_enc        = cached.get("gk_enc")
                        self._gk_clf        = cached.get("gk_clf")
                        self._price_scaler  = cached.get("price_scaler")
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
            from engine.config import get_third_tag_col, COL_GK, COL_NAME, COL_DESCRIPTION, COL_INPUT_CATEGORY
            
            target_col = get_third_tag_col(self.domain)
            missing = {"Name", "basictype", target_col, COL_GK} - set(df.columns)
            if missing:
                logger.warning(f"[TRAIN] ⚠ Catalog missing columns: {missing} — using zero-shot")
                return

            df = df[df["basictype"].str.strip() != ""]
            df = df[df[target_col].str.strip() != ""]
            df = df[df[COL_GK].str.strip() != ""]

            if len(df) < 10:
                logger.warning(f"[TRAIN] ⚠ Only {len(df)} labeled rows — need ≥10. Using zero-shot.")
                return

            # Build query strings exactly matching inference (weighted multi-field embedding)
            from engine.config import CLASSIFIER_WEIGHTS
            names_list = df[COL_NAME].astype(str).str.strip().tolist()
            descs_list = df[COL_DESCRIPTION].astype(str).str.strip().tolist() if COL_DESCRIPTION in df.columns else [""] * len(df)
            col_cat = COL_INPUT_CATEGORY if COL_INPUT_CATEGORY in df.columns else ("category" if "category" in df.columns else "")
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
                self._bt_clf = LogisticRegression(
                    max_iter=1000, C=5.0
                ).fit(X, y_bt)

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
            gk_raw = [[tag.strip() for tag in tags.split(",") if tag.strip()] for tags in df[COL_GK]]
            self._gk_enc = MultiLabelBinarizer()
            y_gk = self._gk_enc.fit_transform(gk_raw)
            base_clf = LogisticRegression(max_iter=250, C=5.0, class_weight="balanced")
            self._gk_clf = OneVsRestClassifier(base_clf).fit(X, y_gk)

            n_bt     = len(set(bt_raw))
            n_third_tag = len(set(third_tag_raw))
            n_gk_tags = len(self._gk_enc.classes_)
            self._trained = True
            logger.info(
                f"[TRAIN] ✓ Trained — "
                f"{len(df)} examples | {n_bt} BTs | {n_third_tag} {target_col.capitalize()}s | {n_gk_tags} GK tags"
            )

            # Save to cache
            os.makedirs(self.cache_dir, exist_ok=True)
            joblib.dump({
                "bt_enc": self._bt_enc,
                "bt_clf": self._bt_clf,
                "third_tag_enc": self._third_tag_enc,
                "third_tag_clf": self._third_tag_clf,
                "gk_enc": self._gk_enc,
                "gk_clf": self._gk_clf,
                "price_scaler": self._price_scaler
            }, cache_file)
            
            if current_hash:
                try:
                    stored_hashes = {}
                    if os.path.exists(HASH_STORAGE_PATH):
                        try:
                            with open(HASH_STORAGE_PATH, "r", encoding="utf-8") as f:
                                stored_hashes = json.load(f)
                        except Exception:
                            stored_hashes = {}
                    stored_hashes[domain_hash_key] = current_hash
                    with open(HASH_STORAGE_PATH, "w", encoding="utf-8") as f:
                        json.dump(stored_hashes, f, indent=4)
                except Exception as hash_err:
                    logger.warning(f"[TRAIN] Failed to write model hash: {hash_err}")

        except Exception as e:
            logger.error(f"[TRAIN] ⚠ Classifier training failed: {e} — using zero-shot", exc_info=True)

    # ── Public API ────────────────────────────────────────────

    def predict_bt(self, vec: np.ndarray, price: Optional[float] = None) -> tuple[str, float, str]:
        """
        Returns (bt_label, confidence, source).
        source is one of: 'trained', 'zero-shot'
        """
        vec_2d = vec.reshape(1, -1) if vec.ndim == 1 else vec

        if self._trained:
            p_val = float(price) if price is not None else 0.0
            scaled_p = self._preprocess_prices([p_val], is_training=False)
            vec_with_price = np.hstack([vec_2d, scaled_p])
            proba = self._bt_clf.predict_proba(vec_with_price)[0]
            best  = int(np.argmax(proba))
            conf  = float(proba[best])
            if conf >= 0.4:
                return self._bt_enc.classes_[best], conf, "trained"

        # Zero-shot: cosine similarity to BT description embeddings
        if not self.bt_labels:
            return "", 0.0, "zero-shot"
            
        scores_pure = (vec_2d @ self.bt_embs_pure.T)[0]
        scores_desc = (vec_2d @ self.bt_embs_desc.T)[0]
        scores = np.maximum(scores_pure, scores_desc)
        best   = int(np.argmax(scores))
        return self.bt_labels[best], float(scores[best]), "zero-shot"

    def batch_predict_bt(self, vecs: np.ndarray, prices: List[float]) -> List[tuple[str, float, str]]:
        """
        Batch version of predict_bt.
        Returns a list of (bt_label, confidence, source) tuples.
        """
        if vecs.shape[0] == 0:
            return []
            
        results = []
        if self._trained:
            scaled_p = self._preprocess_prices(prices, is_training=False)
            vec_with_price = np.hstack([vecs, scaled_p])
            probas = self._bt_clf.predict_proba(vec_with_price)
            bests = np.argmax(probas, axis=1)
            confs = np.max(probas, axis=1)
            
            # For each item, decide if trained model is confident enough, else fallback to zero-shot
            zero_shot_indices = []
            for i in range(len(vecs)):
                if confs[i] >= 0.4:
                    results.append((self._bt_enc.classes_[bests[i]], float(confs[i]), "trained"))
                else:
                    results.append(None) # Placeholder for zero-shot
                    zero_shot_indices.append(i)
        else:
            results = [None] * len(vecs)
            zero_shot_indices = list(range(len(vecs)))

        if zero_shot_indices:
            if not self.bt_labels:
                for i in zero_shot_indices:
                    results[i] = ("", 0.0, "zero-shot")
            else:
                zs_vecs = vecs[zero_shot_indices]
                scores_pure = zs_vecs @ self.bt_embs_pure.T
                scores_desc = zs_vecs @ self.bt_embs_desc.T
                scores = np.maximum(scores_pure, scores_desc)
                bests = np.argmax(scores, axis=1)
                confs = np.max(scores, axis=1)
                for idx_in_zs, orig_idx in enumerate(zero_shot_indices):
                    results[orig_idx] = (self.bt_labels[bests[idx_in_zs]], float(confs[idx_in_zs]), "zero-shot")
                    
        return results

    def predict_gk(self, vec: np.ndarray, price: Optional[float] = None) -> tuple[list[str], float, str]:
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
        threshold = 0.5
        predicted_indices = np.where(proba >= threshold)[0]
        
        if len(predicted_indices) == 0:
            return [], 0.0, "zero-shot"
            
        tags = self._gk_enc.classes_[predicted_indices].tolist()
        conf = float(np.mean(proba[predicted_indices]))
        return tags, conf, "trained"

    def batch_predict_gk(self, vecs: np.ndarray, prices: List[float]) -> List[tuple[list[str], float, str]]:
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
        threshold = 0.5

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
        self, vec: np.ndarray, name: str, description: str = "", predicted_bt: str = "", price: Optional[float] = None
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
            best  = int(np.argmax(proba))
            conf  = float(proba[best])
            if conf >= 0.4:
                return self._third_tag_enc.classes_[best], conf, "trained"

        # Zero-shot
        if not self.third_tag_labels:
            return "", 0.0, "zero-shot"

        scores_pure = (vec_2d @ self.third_tag_embs_pure.T)[0]
        scores_desc = (vec_2d @ self.third_tag_embs_desc.T)[0]
        scores = np.maximum(scores_pure, scores_desc)
        best   = int(np.argmax(scores))
        return self.third_tag_labels[best], float(scores[best]), "zero-shot"

    def batch_predict_third_tag(
        self,
        vecs: np.ndarray,
        names: List[str] = None,
        descriptions: List[str] = None,
        predicted_bts: List[str] = None,
        prices: List[float] = None,
    ) -> List[tuple[str, float, str]]:
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

        results: List[Optional[Tuple[str, float, str]]] = [None] * n
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
                    results[orig_idx] = (self._third_tag_enc.classes_[bests[list_idx]], float(confs[list_idx]), "trained")
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
                    results[orig_idx] = (self.third_tag_labels[bests[list_idx]], float(confs[list_idx]), "zero-shot")

        return results

    def get_guaranteed_gk(self, bt: str) -> list[str]:
        """
        Return umbrella GK tags for a BT.
        e.g. bt='Iced Coffee' → ['Iced Coffee', 'Beverage', 'Coffee']
        These are tags the semantic GK search cannot reliably find on its own.
        """
        return self.bt_to_gk_umbrella.get(bt, [])
