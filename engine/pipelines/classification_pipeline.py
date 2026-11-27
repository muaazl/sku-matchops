import functools
import hashlib
import json
import logging
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from engine.core import config
from engine.core.config import (
    ALPHA,
    AUTO_THRESHOLD,
    COL_DESCRIPTION,
    COL_GK,
    COL_NAME,
    FUSION_METHOD,
    REVIEW_THRESHOLD,
    RRF_K,
    TAG_SEARCH_LIMIT,
    USE_RERANKER,
    get_third_tag_col,
    resolve_input_category_col,
)
from engine.core.config import (
    TOP_K_RETRIEVAL as TOP_K_FUSED,
)
from engine.inference.arcface_model import ArcFaceOnnxModel
from engine.inference.logreg_model import LogregClassifierModel
from engine.pipelines.cold_start_pipeline import ColdStartRouter
from engine.preprocessing.flavor_preprocessor import (
    build_food_flavors_info,
    extract_canonical_flavors,
)
from engine.preprocessing.text_normalizer import build_ner_input_text
from engine.preprocessing.weight_preprocessor import calculate_dataframe_hash
from engine.vector_store.vector_store import VectorStore

logger = logging.getLogger("matchops.classifier")


class TaxonomyClassifier:
    """
    Classifies SKU Basic Type and a third tag (Region/Category) using active embedding vectors.
    """

    @property
    def cache_dir(self) -> Path:
        return getattr(self, "_cache_dir", Path(config.CACHE_DIR))

    @cache_dir.setter
    def cache_dir(self, value) -> None:
        self._cache_dir = Path(value or config.CACHE_DIR)

    def _get_arcface_model(self) -> ArcFaceOnnxModel:
        model = getattr(self, "_arcface_model", None)
        if model is None:
            model = ArcFaceOnnxModel()
            self._arcface_model = model
        return model

    def _get_logreg_model(self) -> LogregClassifierModel:
        model = getattr(self, "_logreg_model", None)
        if model is None:
            model = LogregClassifierModel()
            self._logreg_model = model
        return model

    @property
    def _arcface_session(self):
        return self._get_arcface_model().session

    @_arcface_session.setter
    def _arcface_session(self, value) -> None:
        self._get_arcface_model().session = value

    @property
    def _arcface_classes(self):
        return self._get_arcface_model().classes

    @_arcface_classes.setter
    def _arcface_classes(self, value) -> None:
        self._get_arcface_model().classes = value

    @property
    def _arcface_price_scaler(self):
        return self._get_arcface_model().price_scaler

    @_arcface_price_scaler.setter
    def _arcface_price_scaler(self, value) -> None:
        self._get_arcface_model().price_scaler = value

    @property
    def _arcface_input_name(self):
        return self._get_arcface_model().input_name

    @_arcface_input_name.setter
    def _arcface_input_name(self, value) -> None:
        self._get_arcface_model().input_name = value

    @property
    def _trained(self) -> bool:
        return self._get_logreg_model().trained

    @_trained.setter
    def _trained(self, value: bool) -> None:
        self._get_logreg_model().trained = bool(value)

    @property
    def _price_scaler(self):
        return self._get_logreg_model().price_scaler

    @_price_scaler.setter
    def _price_scaler(self, value) -> None:
        self._get_logreg_model().price_scaler = value

    @property
    def _bt_enc(self):
        return self._get_logreg_model().bt_enc

    @_bt_enc.setter
    def _bt_enc(self, value) -> None:
        self._get_logreg_model().bt_enc = value

    @property
    def _bt_clf(self):
        return self._get_logreg_model().bt_clf

    @_bt_clf.setter
    def _bt_clf(self, value) -> None:
        self._get_logreg_model().bt_clf = value

    @property
    def _third_tag_enc(self):
        return self._get_logreg_model().third_tag_enc

    @_third_tag_enc.setter
    def _third_tag_enc(self, value) -> None:
        self._get_logreg_model().third_tag_enc = value

    @property
    def _third_tag_clf(self):
        return self._get_logreg_model().third_tag_clf

    @_third_tag_clf.setter
    def _third_tag_clf(self, value) -> None:
        self._get_logreg_model().third_tag_clf = value

    @property
    def _gk_enc(self):
        return self._get_logreg_model().gk_enc

    @_gk_enc.setter
    def _gk_enc(self, value) -> None:
        self._get_logreg_model().gk_enc = value

    @property
    def _gk_clf(self):
        return self._get_logreg_model().gk_clf

    @_gk_clf.setter
    def _gk_clf(self, value) -> None:
        self._get_logreg_model().gk_clf = value

    def __init__(
        self,
        model,
        domain: str,
        descriptions: dict,
        cache_dir: Path | str | None = None,
        cat_df: pd.DataFrame | None = None,
        brands_df: pd.DataFrame | None = None,
        force_retrain: bool = False,
        reranker=None,
    ):
        self.model = model
        self.reranker = reranker
        self.domain = domain
        self.cache_dir = Path(cache_dir or config.CACHE_DIR)
        self._arcface_model = ArcFaceOnnxModel()
        self._logreg_model = LogregClassifierModel()
        self.cat_df = cat_df if cat_df is not None else pd.DataFrame()
        self.brands_df = brands_df if brands_df is not None else pd.DataFrame()
        self.food_flavors_dict, _, _, _, _ = build_food_flavors_info(brands_df)

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

        self.bt_model = config.get_bt_model(domain)
        if self.bt_model == "arcface":
            self._load_arcface_model()

        self._try_train(force_retrain=force_retrain)

        self.cold_start_router = None
        try:
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
                rerank_engine=self.reranker,
            )
            logger.info(
                f"[{domain.upper()}] ColdStartRouter initialized successfully as classifier fallback."
            )
        except Exception as router_err:
            logger.warning(
                f"[{domain.upper()}] ColdStartRouter initialization failed: {router_err}"
            )
        self.cat_df = pd.DataFrame()
        self.brands_df = pd.DataFrame()

    def _load_arcface_model(self) -> None:
        """Loads the INT8 ArcFace ONNX model and label mapping for the domain."""
        self._get_arcface_model().load_arcface_artifacts(self.domain)

    def _preprocess_arcface_prices(self, prices_list: list[float]) -> np.ndarray:
        """Preprocesses prices using the ArcFace model's fitted StandardScaler."""
        return self._get_arcface_model().preprocess_prices(prices_list)

    def _embed_cached(self, texts: list[str], cache_key: str) -> np.ndarray:
        """Embed a list of texts with disk caching, keyed by text content and active EMBEDDING_MODEL."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = hashlib.sha256(
            f"{config.EMBEDDING_MODEL}||{'|'.join(texts)}".encode()
        ).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}_{fingerprint[:12]}.pkl"

        if cache_file.exists():
            try:
                cached = joblib.load(cache_file)
                if cached.get("fingerprint") == fingerprint and cached.get("embedding_model") in (
                    None,
                    config.EMBEDDING_MODEL,
                ):
                    return cached["embeddings"]
            except Exception:
                pass

        out = self.model.encode(texts)
        embs = out["dense"]
        joblib.dump(
            {
                "fingerprint": fingerprint,
                "embedding_model": config.EMBEDDING_MODEL,
                "embeddings": embs,
            },
            cache_file,
        )
        return embs

    def _embed_weighted_sku_incremental(
        self,
        names_list: list[str],
        descs_list: list[str],
        cats_list: list[str],
    ) -> np.ndarray:
        """
        Embed training SKUs incrementally, caching the computed dense vectors.
        """
        if not names_list:
            return np.empty((0, config.EMBEDDING_DIM))

        cache_file = self.cache_dir / f"{self.domain}_weighted_skus_cache.pkl"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        sku_cache = {}
        if cache_file.exists():
            try:
                loaded_cache = joblib.load(cache_file)
                cached_model = (
                    loaded_cache.get("__embedding_model__")
                    if isinstance(loaded_cache, dict)
                    else None
                )
                has_model_marker = (self.cache_dir / "qdrant_embedding_models.json").exists() or (
                    self.cache_dir / "active_embedding_model.txt"
                ).exists()
                if isinstance(loaded_cache, dict) and (
                    cached_model == config.EMBEDDING_MODEL
                    or (cached_model is None and not has_model_marker)
                ):
                    sku_cache = loaded_cache
            except Exception:
                sku_cache = {}

        keys = [
            f"{n.strip()}||{d.strip()}||{c.strip()}"
            for n, d, c in zip(names_list, descs_list, cats_list)
        ]

        missing_indices = [i for i, key in enumerate(keys) if key not in sku_cache]

        if missing_indices:
            logger.info(
                f"[EMBED] {self.domain} classifier training: {len(missing_indices)} items to encode "
                f"({len(keys) - len(missing_indices)} cached)."
            )
            missing_names = [names_list[i] for i in missing_indices]
            missing_descs = [descs_list[i] for i in missing_indices]
            missing_cats = [cats_list[i] for i in missing_indices]

            embs = self.model.embed_weighted_sku(
                missing_names, missing_descs, missing_cats, weights=config.CLASSIFIER_WEIGHTS
            )
            dense_vectors = embs["dense"]

            for idx, i in enumerate(missing_indices):
                key = keys[i]
                sku_cache[key] = dense_vectors[idx]

            sku_cache["__embedding_model__"] = config.EMBEDDING_MODEL
            joblib.dump(sku_cache, cache_file)
            logger.info(f"[EMBED] {self.domain} classifier training: cache updated.")
        else:
            logger.info(
                f"[EMBED] {self.domain} classifier training: all {len(keys)} training embeddings served from cache."
            )

        return np.vstack([sku_cache[key] for key in keys])

    def _preprocess_prices(self, prices_list: list[float], is_training: bool = False) -> np.ndarray:
        """Log-scales prices and applies/fits StandardScaler."""
        return self._get_logreg_model().preprocess_prices(prices_list, is_training=is_training)

    def _try_train(self, force_retrain: bool = False) -> None:
        if self.cat_df.empty:
            logger.warning("[TRAIN] Empty catalog dataframe — using zero-shot mode")
            return

        try:
            from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

            cache_file = self.cache_dir / f"{self.domain}_classifier_model.pkl"
            hash_storage_path = Path(config.CACHE_DIR) / "model_hashes.json"
            domain_hash_key = f"{self.domain}_training_state"
            logreg = self._get_logreg_model()

            old_models = {}
            cached = logreg.load_cached_models(cache_file, config.EMBEDDING_MODEL)
            if cached:
                if not force_retrain:
                    logreg.apply_cached_state(cached)
                    logger.info(f"[TRAIN] Loaded classifier model from cache ({self.domain}).")
                    return
                old_models = cached

            current_hash = calculate_dataframe_hash(self.cat_df, domain=self.domain)
            logger.info(f"[TRAIN] Training classifier model for {self.domain}...")

            df = self.cat_df.fillna("")
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

            names_list = df[COL_NAME].astype(str).str.strip().tolist()
            descs_list = (
                df[COL_DESCRIPTION].astype(str).str.strip().tolist()
                if COL_DESCRIPTION in df.columns
                else [""] * len(df)
            )
            col_cat = resolve_input_category_col(df.columns)
            cats_list = df[col_cat].astype(str).str.strip().tolist() if col_cat else [""] * len(df)

            logger.info(f"[TRAIN] Training on {len(df)} labeled SKUs...")
            X = self._embed_weighted_sku_incremental(names_list, descs_list, cats_list)

            prices_list = pd.to_numeric(df["Price"], errors="coerce").fillna(0.0).tolist()
            scaled_prices = self._preprocess_prices(prices_list, is_training=True)
            X = np.hstack([X, scaled_prices])

            bt_raw = df["basictype"].str.strip().tolist()
            logreg.bt_enc = LabelEncoder().fit(bt_raw)
            y_bt = logreg.bt_enc.transform(bt_raw)

            third_tag_raw = df[target_col].str.strip().tolist()
            logreg.third_tag_enc = LabelEncoder().fit(third_tag_raw)
            y_third_tag = logreg.third_tag_enc.transform(third_tag_raw)

            gk_raw = [
                [tag.strip() for tag in tags.split(",") if tag.strip()] for tags in df[COL_GK]
            ]
            logreg.gk_enc = MultiLabelBinarizer()
            y_gk = logreg.gk_enc.fit_transform(gk_raw)

            logreg.fit_classifiers(
                X=X,
                y_bt=y_bt,
                y_third_tag=y_third_tag,
                y_gk=y_gk,
                domain=self.domain,
                target_col=target_col,
                old_models=old_models,
            )

            n_bt = len(set(bt_raw))
            n_third_tag = len(set(third_tag_raw))
            n_gk_tags = len(logreg.gk_enc.classes_)
            logger.info(
                f"[TRAIN] ✓ Trained — "
                f"{len(df)} examples | {n_bt} BTs | {n_third_tag} {target_col.capitalize()}s | {n_gk_tags} GK tags"
            )

            logreg.save_cached_models(cache_file, config.EMBEDDING_MODEL)

            if current_hash:
                try:
                    stored_hashes = {}
                    if hash_storage_path.exists():
                        try:
                            with hash_storage_path.open(encoding="utf-8") as f:
                                stored_hashes = json.load(f)
                        except Exception:
                            stored_hashes = {}
                    stored_hashes[domain_hash_key] = current_hash
                    with hash_storage_path.open("w", encoding="utf-8") as f:
                        json.dump(stored_hashes, f, indent=4)
                except Exception as hash_err:
                    logger.warning(f"[TRAIN] Failed to write model hash: {hash_err}")

        except Exception as e:
            logger.error(
                f"[TRAIN] ⚠ Classifier training failed: {e} — using zero-shot", exc_info=True
            )

    @property
    def active_bt_model(self) -> str:
        """Returns the active BasicType classification model identifier ('arcface' or 'logreg')."""
        return getattr(self, "bt_model", "logreg")

    def register_new_tag(self, tag: str, description: str = "") -> None:
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
        """Returns (bt_label, confidence, source, propagated_gks)."""
        vec_2d = vec.reshape(1, -1) if vec.ndim == 1 else vec
        p_val = float(price) if price is not None else 0.0
        cold_router = getattr(self, "cold_start_router", None)

        arcface_model = self._get_arcface_model()
        if self.bt_model == "arcface" and arcface_model.session is not None:
            label, conf = arcface_model.predict_single(vec_2d, p_val)
            if conf >= config.BT_TRAINED_CONFIDENCE_THRESHOLD:
                return label, conf, "trained", []
        elif self.bt_model == "logreg" and self._trained and self._bt_clf is not None:
            bests, confs = self._get_logreg_model().predict_bt_batch(vec_2d, [p_val])
            conf = float(confs[0])
            if conf >= config.BT_TRAINED_CONFIDENCE_THRESHOLD:
                return self._bt_enc.classes_[int(bests[0])], conf, "trained", []

        if cold_router is not None:
            r_tag, r_conf, r_src, r_gks = cold_router.route_single(
                vec_2d[0], sku_name=sku_name, sku_description=sku_description, price=price
            )
            if r_tag and r_conf >= config.BT_ZERO_SHOT_CONFIDENCE_THRESHOLD:
                return r_tag, r_conf, r_src, r_gks

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
        """Batch version of predict_bt."""
        if vecs.shape[0] == 0:
            return []

        n = len(vecs)
        results: list[tuple[str, float, str, list[str]] | None] = [None] * n

        if sku_names is None:
            sku_names = [""] * n
        if sku_descriptions is None:
            sku_descriptions = [""] * n

        zero_shot_indices = []

        arcface_model = self._get_arcface_model()
        if self.bt_model == "arcface" and arcface_model.session is not None:
            bests, confs = arcface_model.predict_batch(vecs, prices)
            for i in range(len(vecs)):
                if confs[i] >= config.BT_TRAINED_CONFIDENCE_THRESHOLD:
                    results[i] = (arcface_model.classes[bests[i]], float(confs[i]), "trained", [])
                else:
                    zero_shot_indices.append(i)
        elif self.bt_model == "logreg" and self._trained and self._bt_clf is not None:
            bests, confs = self._get_logreg_model().predict_bt_batch(vecs, prices)
            for i in range(len(vecs)):
                if confs[i] >= config.BT_TRAINED_CONFIDENCE_THRESHOLD:
                    results[i] = (self._bt_enc.classes_[bests[i]], float(confs[i]), "trained", [])
                else:
                    zero_shot_indices.append(i)
        else:
            zero_shot_indices = list(range(len(vecs)))

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

        return [r for r in results if r is not None]

    def predict_gk(
        self, vec: np.ndarray, price: float | None = None
    ) -> tuple[list[str], float, str]:
        """Returns (list_of_gk_tags, confidence, source)."""
        if not self._trained or getattr(self, "_gk_clf", None) is None:
            return [], 0.0, "zero-shot"

        vec_2d = vec.reshape(1, -1) if vec.ndim == 1 else vec
        p_val = float(price) if price is not None else 0.0
        scaled_p = self._preprocess_prices([p_val], is_training=False)
        vec_with_price = np.hstack([vec_2d, scaled_p])

        proba = self._gk_clf.predict_proba(vec_with_price)[0]
        threshold = config.GK_TRAINED_CONFIDENCE_THRESHOLD
        predicted_indices = np.nonzero(proba >= threshold)[0]

        if len(predicted_indices) == 0:
            return [], 0.0, "zero-shot"

        tags = self._gk_enc.classes_[predicted_indices].tolist()
        conf = float(np.mean(proba[predicted_indices]))
        return tags, conf, "trained"

    def batch_predict_gk(
        self, vecs: np.ndarray, prices: list[float]
    ) -> list[tuple[list[str], float, str]]:
        """Batch version of predict_gk."""
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
            predicted_indices = np.nonzero(proba >= threshold)[0]
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
        """Returns (third_tag_label, confidence, source)."""
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
            if conf >= config.BT_TRAINED_CONFIDENCE_THRESHOLD:
                return self._third_tag_enc.classes_[best], conf, "trained"

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
        names: list[str] | None = None,
        descriptions: list[str] | None = None,
        predicted_bts: list[str] | None = None,
        prices: list[float] | None = None,
    ) -> list[tuple[str, float, str]]:
        """Batch version of predict_third_tag."""
        n = len(vecs)
        if n == 0:
            return []

        if predicted_bts is None:
            predicted_bts = [""] * n
        if prices is None:
            prices = [0.0] * n

        results: list[tuple[str, float, str] | None] = [None] * n
        remaining_indices = []

        for i in range(n):
            bt = predicted_bts[i]
            if bt and bt in self.third_tag_overrides:
                results[i] = (self.third_tag_overrides[bt], 1.0, "override")
            else:
                remaining_indices.append(i)

        if not remaining_indices:
            return [r for r in results if r is not None]

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
                if confs[list_idx] >= config.BT_TRAINED_CONFIDENCE_THRESHOLD:
                    results[orig_idx] = (
                        self._third_tag_enc.classes_[bests[list_idx]],
                        float(confs[list_idx]),
                        "trained",
                    )
                else:
                    zero_shot_indices.append(orig_idx)
        else:
            zero_shot_indices = remaining_indices

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

        return [r for r in results if r is not None]

    def get_guaranteed_gk(self, bt: str) -> list[str]:
        """Return umbrella GK tags for a BT."""
        return self.bt_to_gk_umbrella.get(bt, [])


def get_status(confidence: float, has_tags: bool, source: str = "") -> str:
    """Resolves classification status (AUTO, REVIEW, LOW) based on confidence and provenance."""
    if not has_tags:
        return "LOW"
    if source in ("override", "keyword"):
        return "AUTO"
    if source == "reranker":
        auto_thr = config.CONFIDENCE_THRESHOLD_HIGH
        review_thr = config.RERANKER_THRESHOLD
    else:
        auto_thr = AUTO_THRESHOLD
        review_thr = REVIEW_THRESHOLD
    if confidence >= auto_thr:
        return "AUTO"
    if confidence >= review_thr:
        return "REVIEW"
    return "LOW"


def resolve_gk_confidence_and_status(
    trained_conf: float,
    trained_source: str,
    final_conf: float,
    used_reranker: bool,
    has_tags: bool,
) -> tuple[float, str, str]:
    """Selects calibrated GK confidence, status, and winning source across trained vs reranked paths."""
    if trained_source == "trained" and trained_conf >= AUTO_THRESHOLD:
        gk_conf = float(trained_conf)
        gk_source = "trained"
    elif used_reranker:
        gk_conf = float(final_conf)
        gk_source = "reranker"
    else:
        gk_conf = float(final_conf)
        gk_source = ""
    return gk_conf, get_status(gk_conf, has_tags, gk_source), gk_source


def compute_weighted_confidence(
    bt_conf: float | None = None,
    gk_conf: float | None = None,
    region_conf: float | None = None,
    has_bt: bool = True,
    has_gk: bool = True,
    has_region: bool = True,
    w_bt: float = 0.5,
    w_gk: float = 0.3,
    w_reg: float = 0.2,
) -> float:
    """Computes dynamically normalized weighted confidence across available tags."""
    weights = []
    confs = []
    if has_bt and bt_conf is not None:
        try:
            confs.append(float(bt_conf))
            weights.append(w_bt)
        except (ValueError, TypeError):
            pass
    if has_gk and gk_conf is not None:
        try:
            confs.append(float(gk_conf))
            weights.append(w_gk)
        except (ValueError, TypeError):
            pass
    if has_region and region_conf is not None:
        try:
            confs.append(float(region_conf))
            weights.append(w_reg)
        except (ValueError, TypeError):
            pass

    if not confs or sum(weights) == 0:
        try:
            return float(bt_conf or 0.0)
        except (ValueError, TypeError):
            return 0.0

    total_w = sum(weights)
    return round(sum(c * (w / total_w) for c, w in zip(confs, weights)), 4)


def _rrf_fusion(
    dense_results,
    bm25_results,
    sparse_results=None,
    k=RRF_K,
) -> list:
    """Reciprocal Rank Fusion across Dense, BM25, and optional BGE-M3 learned sparse hits."""
    scores: dict = {}
    for rank_list in (dense_results, bm25_results, sparse_results):
        if not rank_list:
            continue
        for rank, hit in enumerate(rank_list):
            tag = hit.payload.get("tag")
            scores[tag] = scores.get(tag, 0.0) + 1.0 / (k + rank + 1)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"tag": tag, "score": score, "source": "rrf"} for tag, score in fused]


@functools.lru_cache(maxsize=128)
def _get_gk_regex(gk_tuple: tuple[str, ...]) -> re.Pattern:
    """Pre-compiles a single regex pattern for a list of GKs, optimized with alternation."""
    sorted_gks = sorted(gk_tuple, key=len, reverse=True)
    escaped_gks = [re.escape(gk) for gk in sorted_gks]
    pattern_str = r"(?<![a-z0-9])(" + "|".join(escaped_gks) + r")(?![a-z0-9])"
    return re.compile(pattern_str)


def _weighted_fusion(dense_results, bm25_results, sparse_results=None, alpha=ALPHA) -> list:
    """Weighted Sum Fusion across Dense, BM25, and optional BGE-M3 learned sparse hits."""
    scores: dict = {}

    for hit in dense_results or ():
        tag = hit.payload.get("tag")
        scores[tag] = scores.get(tag, 0) + alpha * hit.score

    lexical_lists = [lst for lst in (bm25_results, sparse_results) if lst]
    if lexical_lists:
        lex_weight = (1 - alpha) / len(lexical_lists)
        for lst in lexical_lists:
            for hit in lst:
                tag = hit.payload.get("tag")
                scores[tag] = scores.get(tag, 0) + lex_weight * hit.score

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"tag": tag, "score": score, "source": "weighted"} for tag, score in fused]


class DomainStrategy:
    @staticmethod
    def filter_search_tags(
        allowed_gks_for_bt: list, trained_conf: float, source: str, guaranteed: list
    ) -> tuple:
        return allowed_gks_for_bt, None

    @staticmethod
    def apply_post_search_filters(top_candidates: list, allowed_gks_lower: set) -> list:
        return top_candidates

    @staticmethod
    def inject_synthetic_tags(
        top_candidates: list, market_brand: str, bt: str, category: str, extracted_flavors: list
    ) -> list:
        return top_candidates

    @staticmethod
    def apply_flavor_leak_filters(
        guaranteed: list,
        trained_gk: list,
        top_candidates: list,
        extracted_flavors: list,
        ner_engine,
    ) -> tuple:
        return guaranteed, trained_gk, top_candidates

    @staticmethod
    def get_reranker_threshold(score: float, tag_lower: str, allowed_gks_lower: set) -> bool:
        return False

    @staticmethod
    def filter_final_trained_gk(trained_gk: list, allowed_gks_lower: set) -> list:
        return trained_gk


class FoodStrategy(DomainStrategy):
    @staticmethod
    def filter_search_tags(
        allowed_gks_for_bt: list, trained_conf: float, source: str, guaranteed: list
    ) -> tuple:
        if not allowed_gks_for_bt:
            return None, (
                guaranteed,
                trained_conf if (source == "trained" and trained_conf >= AUTO_THRESHOLD) else 0.0,
            )
        return allowed_gks_for_bt, None

    @staticmethod
    def apply_post_search_filters(top_candidates: list, allowed_gks_lower: set) -> list:
        if allowed_gks_lower:
            return [c for c in top_candidates if c["tag"].lower().strip() in allowed_gks_lower]
        return top_candidates

    @staticmethod
    def inject_synthetic_tags(
        top_candidates: list, market_brand: str, bt: str, category: str, extracted_flavors: list
    ) -> list:
        if extracted_flavors:
            synthetic_tags = list(extracted_flavors)
            if len(extracted_flavors) > 1:
                synthetic_tags.append(" & ".join(extracted_flavors))

            existing_tags_lower = {c["tag"].lower().strip() for c in top_candidates}
            for st in synthetic_tags:
                if st.lower().strip() not in existing_tags_lower:
                    top_candidates.insert(
                        0, {"tag": st.title(), "score": 1.0, "source": "synthetic"}
                    )
        return top_candidates

    @staticmethod
    def apply_flavor_leak_filters(
        guaranteed: list,
        trained_gk: list,
        top_candidates: list,
        extracted_flavors: list,
        ner_engine,
    ) -> tuple:
        if ner_engine:
            input_flavors_canonical = {f.lower().strip() for f in (extracted_flavors or [])}

            def has_unrelated_flavor(tag: str) -> bool:
                gk_strong, gk_weak = ner_engine._get_dict_entities(tag)
                gk_flavors = gk_strong | gk_weak
                return bool(gk_flavors and not gk_flavors.issubset(input_flavors_canonical))

            guaranteed = [tag for tag in guaranteed if not has_unrelated_flavor(tag)]
            trained_gk = [tag for tag in trained_gk if not has_unrelated_flavor(tag)]
            top_candidates = [c for c in top_candidates if not has_unrelated_flavor(c["tag"])]
        return guaranteed, trained_gk, top_candidates

    @staticmethod
    def get_reranker_threshold(score: float, tag_lower: str, allowed_gks_lower: set) -> bool:
        return tag_lower in allowed_gks_lower and score >= config.RERANKER_THRESHOLD

    @staticmethod
    def filter_final_trained_gk(trained_gk: list, allowed_gks_lower: set) -> list:
        if allowed_gks_lower:
            return [t for t in trained_gk if t.lower().strip() in allowed_gks_lower]
        return trained_gk


class MarketStrategy(DomainStrategy):
    @staticmethod
    def filter_search_tags(
        allowed_gks_for_bt: list, trained_conf: float, source: str, guaranteed: list
    ) -> tuple:
        if allowed_gks_for_bt:
            return allowed_gks_for_bt, None
        return None, None

    @staticmethod
    def apply_post_search_filters(top_candidates: list, allowed_gks_lower: set) -> list:
        if allowed_gks_lower:
            if getattr(config, "ALLOW_UNREGISTERED_TEMPLATE_KEYWORDS", True):
                return [
                    c
                    for c in top_candidates
                    if c["tag"].lower().strip() in allowed_gks_lower
                    or c.get("source") == "synthetic"
                ]
            return [c for c in top_candidates if c["tag"].lower().strip() in allowed_gks_lower]
        return top_candidates

    @staticmethod
    def inject_synthetic_tags(
        top_candidates: list, market_brand: str, bt: str, category: str, extracted_flavors: list
    ) -> list:
        if market_brand:
            synthetic_tags = []
            if bt:
                synthetic_tags.append(f"{market_brand} {bt}")
            if category:
                synthetic_tags.append(f"{market_brand} {category}")

            existing_tags_lower = {c["tag"].lower().strip() for c in top_candidates}
            for st in synthetic_tags:
                if st.lower().strip() not in existing_tags_lower:
                    top_candidates.insert(0, {"tag": st, "score": 1.0, "source": "synthetic"})
        return top_candidates

    @staticmethod
    def get_reranker_threshold(score: float, tag_lower: str, allowed_gks_lower: set) -> bool:
        if allowed_gks_lower:
            return tag_lower in allowed_gks_lower and score >= config.RERANKER_THRESHOLD
        return score >= config.RERANKER_THRESHOLD

    @staticmethod
    def filter_final_trained_gk(trained_gk: list, allowed_gks_lower: set) -> list:
        if allowed_gks_lower:
            return [t for t in trained_gk if t.lower().strip() in allowed_gks_lower]
        return trained_gk


def get_strategy(domain: str) -> DomainStrategy:
    if domain == "food":
        return FoodStrategy()
    if domain == "market":
        return MarketStrategy()
    return DomainStrategy()


def tag_all_skus(
    sku_names,
    sku_categories,
    query_embeddings,
    vector_store,
    reranker,
    classifier,
    sku_descriptions=None,
    sku_prices=None,
    embed_engine=None,
    ner_engine=None,
    is_cancelled=None,
    progress_callback=None,
):
    n = len(sku_names)
    if n == 0:
        return []

    if sku_descriptions is None:
        sku_descriptions = [""] * n
    if sku_prices is None:
        sku_prices = [0.0] * n
    if sku_categories is None:
        sku_categories = [""] * n

    brands_per_sku = [""] * n
    flavors_per_sku: list[list[str]] = [[] for _ in range(n)]

    if ner_engine:
        combined_texts = []
        for idx_sku in range(n):
            name_txt = sku_names[idx_sku] or ""
            desc_txt = sku_descriptions[idx_sku] if idx_sku < len(sku_descriptions) else ""
            cat_txt = sku_categories[idx_sku] if idx_sku < len(sku_categories) else ""
            combined_texts.append(build_ner_input_text(name_txt, desc_txt, cat_txt))

        if classifier.domain == "market":
            ner_results = ner_engine.batch_extract_entities(combined_texts)
            for i, ents in enumerate(ner_results):
                brand_set = ents.get("brand", set())
                if brand_set:
                    brands_per_sku[i] = sorted(brand_set, key=len, reverse=True)[0].title()
        elif classifier.domain == "food":
            ner_results = ner_engine.batch_extract_entities(combined_texts)
            for i, ents in enumerate(ner_results):
                flavor_set = ents.get("flavor", set())
                if flavor_set:
                    flavors_per_sku[i] = list(flavor_set)

    if is_cancelled and is_cancelled():
        raise InterruptedError("Job cancelled")
    if progress_callback:
        progress_callback(15.0)

    if isinstance(query_embeddings, dict):
        dense_vecs = np.asarray(query_embeddings["dense"])
        bm25_vecs = query_embeddings.get("bm25") or [{} for _ in range(len(dense_vecs))]
        sparse_vecs = query_embeddings.get("sparse") or []
    else:
        dense_vecs = np.array([q["dense"] for q in query_embeddings])
        bm25_vecs = [q.get("bm25", {}) for q in query_embeddings]
        raw_sp = [q.get("sparse") for q in query_embeddings]
        sparse_vecs = [s or {} for s in raw_sp] if any(s for s in raw_sp) else []

    raw_bt_preds = classifier.batch_predict_bt(
        dense_vecs, sku_prices, sku_names=sku_names, sku_descriptions=sku_descriptions
    )
    bt_results = [("", 0.0, "")] * n
    router_gks = {}
    for i in range(n):
        bt_tag, confidence, source, gks = raw_bt_preds[i]
        if gks:
            router_gks[i] = gks
        if confidence < REVIEW_THRESHOLD:
            bt_results[i] = ("", 0.0, source)
            continue
        extracted_flavors = flavors_per_sku[i]
        if extracted_flavors and bt_tag:
            bt_flavors = extract_canonical_flavors(bt_tag, classifier.food_flavors_dict)
            input_flavors = set()
            for f in extracted_flavors:
                input_flavors.update(extract_canonical_flavors(f, classifier.food_flavors_dict))
            if bt_flavors and input_flavors and bt_flavors.isdisjoint(input_flavors):
                bt_results[i] = ("", 0.0, "conflict")
                continue
        bt_results[i] = (bt_tag, confidence, source)

    predicted_bts = [b[0] for b in bt_results]

    if is_cancelled and is_cancelled():
        raise InterruptedError("Job cancelled")
    if progress_callback:
        progress_callback(30.0)

    raw_third_tag_preds = classifier.batch_predict_third_tag(
        dense_vecs, sku_names, sku_descriptions, predicted_bts, sku_prices
    )
    third_tag_results = [(tag, conf, source) for tag, conf, source in raw_third_tag_preds]

    trained_gk_preds = classifier.batch_predict_gk(dense_vecs, sku_prices)

    if is_cancelled and is_cancelled():
        raise InterruptedError("Job cancelled")
    if progress_callback:
        progress_callback(45.0)

    strategy = get_strategy(classifier.domain)
    bt_gk_map = getattr(classifier, "bt_gk_map", {})

    all_guaranteed = []
    all_allowed_gks_lower = []
    all_search_allowed_tags = []
    needs_search_indices = []
    early_return_results = {}

    for i in range(n):
        bt = predicted_bts[i]
        guaranteed = classifier.get_guaranteed_gk(bt)
        if not isinstance(guaranteed, list):
            guaranteed = list(guaranteed)

        if i in router_gks:
            for extra_gk in router_gks[i]:
                if extra_gk not in guaranteed:
                    guaranteed.append(extra_gk)

        trained_gk, trained_conf, trained_source = trained_gk_preds[i]
        allowed_gks_for_bt = bt_gk_map.get(bt, [])
        allowed_gks_lower = {kw.lower().strip() for kw in allowed_gks_for_bt}

        if bt and bt not in guaranteed:
            bt_lower = bt.lower().strip()
            all_known_gks_lower: set = set()
            for gk_list in bt_gk_map.values():
                for gk in gk_list:
                    all_known_gks_lower.add(gk.lower().strip())
            if bt_lower in all_known_gks_lower:
                guaranteed.append(bt)

        sku_name_lower = (sku_names[i] or "").lower()
        if allowed_gks_for_bt:
            gk_pool = [gk.lower().strip() for gk in allowed_gks_for_bt if gk.strip()]
            gk_map = {gk.lower().strip(): gk for gk in allowed_gks_for_bt if gk.strip()}
        else:
            gk_map = {}
            for gk_list in bt_gk_map.values():
                for gk in gk_list:
                    clean = gk.lower().strip()
                    if clean and clean not in gk_map:
                        gk_map[clean] = gk
            gk_pool = list(gk_map.keys())

        if gk_pool:
            pattern = _get_gk_regex(tuple(gk_pool))
            for match in pattern.finditer(sku_name_lower):
                matched_gk_lower = match.group(1)
                original_gk = gk_map[matched_gk_lower]
                if original_gk not in guaranteed:
                    guaranteed.append(original_gk)

        search_allowed_tags, early_return = strategy.filter_search_tags(
            allowed_gks_for_bt, trained_conf, trained_source, guaranteed
        )
        all_guaranteed.append(guaranteed)
        all_allowed_gks_lower.append(allowed_gks_lower)
        all_search_allowed_tags.append(search_allowed_tags)

        if early_return is not None:
            early_return_results[i] = early_return
        else:
            needs_search_indices.append(i)

    gk_final_results = {}
    for i, early_res in early_return_results.items():
        _, tr_conf, tr_src = trained_gk_preds[i]
        early_conf, early_status, _ = resolve_gk_confidence_and_status(
            tr_conf, tr_src, early_res[1], False, bool(early_res[0])
        )
        gk_final_results[i] = (early_res[0], early_conf, early_status)

    if needs_search_indices:
        search_dense = [dense_vecs[i] for i in needs_search_indices]
        search_bm25 = [bm25_vecs[i] if i < len(bm25_vecs) else {} for i in needs_search_indices]
        search_sparse = [sparse_vecs[i] for i in needs_search_indices] if sparse_vecs else None
        search_allowed = [all_search_allowed_tags[i] for i in needs_search_indices]

        batch_search_hits = vector_store.search_tags_hybrid_batch(
            dense_queries=search_dense,
            sparse_queries=search_sparse,
            filter_dict_type="gk",
            domain=classifier.domain,
            allowed_tags_list=search_allowed,
            limit=TAG_SEARCH_LIMIT,
            bm25_queries=search_bm25,
        )

        if is_cancelled and is_cancelled():
            raise InterruptedError("Job cancelled")
        if progress_callback:
            progress_callback(70.0)

        all_rerank_pairs = []
        unique_pair_idx: dict[tuple[str, str], int] = {}
        pair_mapping = []
        per_item_top_candidates = []

        for list_idx, orig_idx in enumerate(needs_search_indices):
            dense_hits, bm25_hits, sparse_hits = batch_search_hits[list_idx]
            if FUSION_METHOD == "rrf":
                fused = _rrf_fusion(dense_hits, bm25_hits, sparse_hits)
            else:
                fused = _weighted_fusion(dense_hits, bm25_hits, sparse_hits)
            top_candidates = fused[:TOP_K_FUSED]
            top_candidates = strategy.apply_post_search_filters(
                top_candidates, all_allowed_gks_lower[orig_idx]
            )
            top_candidates = strategy.inject_synthetic_tags(
                top_candidates,
                brands_per_sku[orig_idx],
                predicted_bts[orig_idx],
                sku_categories[orig_idx],
                flavors_per_sku[orig_idx],
            )
            ner_eng = getattr(classifier, "ner_engine", ner_engine)
            guaranteed, trained_gk, top_candidates = strategy.apply_flavor_leak_filters(
                all_guaranteed[orig_idx],
                trained_gk_preds[orig_idx][0],
                top_candidates,
                flavors_per_sku[orig_idx],
                ner_eng,
            )
            all_guaranteed[orig_idx] = guaranteed
            trained_gk_preds[orig_idx] = (
                trained_gk,
                trained_gk_preds[orig_idx][1],
                trained_gk_preds[orig_idx][2],
            )
            per_item_top_candidates.append(top_candidates)

            if top_candidates and USE_RERANKER and reranker is not None:
                query_text = (sku_names[orig_idx] + " " + sku_descriptions[orig_idx]).strip()
                for c_idx, c in enumerate(top_candidates):
                    tag_text = c["tag"]
                    pair_key = (query_text, tag_text)
                    p_idx = unique_pair_idx.get(pair_key)
                    if p_idx is None:
                        p_idx = len(all_rerank_pairs)
                        unique_pair_idx[pair_key] = p_idx
                        all_rerank_pairs.append([query_text, tag_text])
                    pair_mapping.append((list_idx, c_idx, p_idx))

        scored_candidates_per_item: dict[int, list[tuple[str, float]]] = {
            list_idx: [] for list_idx in range(len(needs_search_indices))
        }
        if all_rerank_pairs:

            def _gk_rerank_cb(done: int, total: int) -> None:
                if is_cancelled and is_cancelled():
                    raise InterruptedError("Job cancelled")
                if progress_callback and total > 0:
                    progress_callback(70.0 + (20.0 * (done / total)))

            try:
                if hasattr(reranker, "predict"):
                    raw_scores = reranker.predict(all_rerank_pairs, progress_callback=_gk_rerank_cb)
                else:
                    raw_scores = reranker.score_cross_encoder(
                        all_rerank_pairs, progress_callback=_gk_rerank_cb
                    )
            except TypeError:
                raw_scores = (
                    reranker.predict(all_rerank_pairs)
                    if hasattr(reranker, "predict")
                    else reranker.score_cross_encoder(all_rerank_pairs)
                )
            if isinstance(raw_scores, float) or (
                hasattr(raw_scores, "item") and raw_scores.ndim == 0
            ):
                scores = [float(raw_scores)]
            else:
                scores = [float(s) for s in raw_scores]

            for list_idx, c_idx, p_idx in pair_mapping:
                orig_idx = needs_search_indices[list_idx]
                c = per_item_top_candidates[list_idx][c_idx]
                score = scores[p_idx]
                tag_lower = c["tag"].lower().strip()
                if strategy.get_reranker_threshold(
                    score, tag_lower, all_allowed_gks_lower[orig_idx]
                ):
                    scored_candidates_per_item[list_idx].append((c["tag"], score))

        if is_cancelled and is_cancelled():
            raise InterruptedError("Job cancelled")
        if progress_callback:
            progress_callback(90.0)

        for list_idx, orig_idx in enumerate(needs_search_indices):
            top_candidates = per_item_top_candidates[list_idx]
            scored_candidates = scored_candidates_per_item[list_idx]
            scored_candidates.sort(key=lambda x: x[1], reverse=True)

            used_reranker = bool(USE_RERANKER and reranker is not None and top_candidates)
            if used_reranker:
                if scored_candidates:
                    top_score = scored_candidates[0][1]
                    margin_cutoff = max(
                        config.RERANKER_THRESHOLD, top_score - config.RERANKER_MARGIN
                    )
                    scored_candidates = [c for c in scored_candidates if c[1] >= margin_cutoff]
                    max_reranked = int(getattr(config, "MAX_RERANKED_GK_TAGS", 5) or 0)
                    if max_reranked > 0:
                        scored_candidates = scored_candidates[:max_reranked]
                reranked_tags = [tag for tag, score in scored_candidates]
                final_conf = scored_candidates[0][1] if scored_candidates else 0.0
            else:
                reranked_tags = [c["tag"] for c in top_candidates]
                final_conf = min(1.0, top_candidates[0]["score"]) if top_candidates else 0.0

            seen = set()
            merged = []
            trained_gk, trained_conf, trained_source = trained_gk_preds[orig_idx]
            trained_gk = strategy.filter_final_trained_gk(
                trained_gk, all_allowed_gks_lower[orig_idx]
            )

            for tag in all_guaranteed[orig_idx] + trained_gk + reranked_tags:
                key = tag.lower().strip()
                if key not in seen:
                    seen.add(key)
                    merged.append(tag)

            gk_conf, gk_status, _ = resolve_gk_confidence_and_status(
                trained_conf, trained_source, final_conf, used_reranker, bool(merged)
            )
            gk_final_results[orig_idx] = (merged, gk_conf, gk_status)

    results = []
    domain = classifier.domain
    active_model = getattr(classifier, "active_bt_model", "logreg")
    tag_key = "suggested_region" if domain == "food" else "suggested_category"
    conf_key = "region_confidence" if domain == "food" else "category_confidence"
    status_key = "region_status" if domain == "food" else "category_status"
    source_key = "region_source" if domain == "food" else "category_source"

    for i in range(n):
        bt_tag, bt_conf, bt_source = bt_results[i]
        gk_tags, gk_conf, gk_status = gk_final_results.get(i, ([], 0.0, "LOW"))
        third_tag_name, third_tag_conf, third_tag_source = third_tag_results[i]

        res = {
            "suggested_gk": ", ".join(gk_tags),
            "gk_confidence": round(gk_conf, 3),
            "gk_status": gk_status,
            "suggested_bt": bt_tag,
            "bt_confidence": round(bt_conf, 3),
            "bt_status": get_status(bt_conf, bool(bt_tag), bt_source),
            "bt_source": bt_source,
            "bt_model": active_model,
            "model": active_model,
            tag_key: third_tag_name,
            conf_key: round(third_tag_conf, 3),
            status_key: get_status(third_tag_conf, bool(third_tag_name), third_tag_source),
            source_key: third_tag_source,
        }
        gk_str = ", ".join(gk_tags)
        third_label = "Region" if domain == "food" else "Category"
        reasoning = (
            f"Classifier ({active_model}): BT='{bt_tag}' ({bt_source}, {round(bt_conf, 3)}), "
            f"GK='{gk_str}' ({round(gk_conf, 3)}), "
            f"{third_label}='{third_tag_name}' ({third_tag_source}, {round(third_tag_conf, 3)})"
        )
        res["reasoning"] = reasoning
        results.append(res)

    if progress_callback:
        progress_callback(100.0)

    return results
