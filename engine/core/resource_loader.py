"""
SKU MatchOps Engine - Model & Resource Loader
Handles loading, warm-up, and singleton caching of ML pipelines, NER, and Classifiers.
"""

import gc
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from engine import config
from engine.classification.classifier import ZeroShotClassifier
from engine.classification.loader import (
    augment_bt_gk_map_with_training,
    build_bt_third_tag_map_from_catalog,
    build_descriptions,
    build_umbrella_from_training,
)
from engine.data_pipeline.cache_manager import CacheManager
from engine.data_pipeline.ingestion import DataIngestion
from engine.data_pipeline.vector_store import VectorStore
from engine.matching.logic_gates import LogicGates
from engine.matching.matcher import SKUMatcher
from engine.nlp.embedding_engine import EmbeddingEngine
from engine.nlp.ner_engine import NEREngine
from engine.rules_engine import clear_flavor_cache, refresh_rules_cache
from engine.rules_engine.loader import start_invalidation_listener

logger = logging.getLogger("matchops.engine.loader")

_pipelines: dict[str, SKUMatcher] = {}
_classifiers: dict[str, ZeroShotClassifier] = {}
_domain_ner_engines: dict[str, NEREngine] = {}
_vector_store = None
_loader_lock = threading.RLock()
_model_load_lock = threading.Lock()
_queue_worker: threading.Thread | None = None

_pipeline_build_locks = {
    config.DOMAIN_MARKET: threading.Lock(),
    config.DOMAIN_FOOD: threading.Lock(),
}
_classifier_build_locks = {
    config.DOMAIN_MARKET: threading.Lock(),
    config.DOMAIN_FOOD: threading.Lock(),
}

_model_statuses = {
    "market": {"pipeline": "idle", "classifier": "idle"},
    "food": {"pipeline": "idle", "classifier": "idle"},
}


def reset_statuses(clear_caches: bool = False) -> None:
    with _loader_lock:
        if clear_caches:
            _pipelines.clear()
            _classifiers.clear()
            _domain_ner_engines.clear()
        for domain in _model_statuses:
            _model_statuses[domain]["pipeline"] = "idle"
            _model_statuses[domain]["classifier"] = "idle"


def are_all_models_ready() -> bool:
    """True when both market and food pipeline + classifier models are ready."""
    return all(
        _model_statuses.get(d, {}).get("pipeline") == "ready"
        and _model_statuses.get(d, {}).get("classifier") == "ready"
        for d in (config.DOMAIN_MARKET, config.DOMAIN_FOOD)
    )


def is_model_loading() -> bool:
    """True while a background warmup/reload is running or any domain build is active."""
    return _model_load_lock.locked() or any(
        st.get("pipeline") == "loading" or st.get("classifier") == "training"
        for st in _model_statuses.values()
    )


# Shared heavy models
_embed_engine = None
_ner_engine = None


def _get_shared_models():
    """Load the embedding & NER models concurrently and reuse across domains."""
    global _embed_engine, _ner_engine
    if _embed_engine is None or _ner_engine is None:
        with _loader_lock:
            if _embed_engine is None or _ner_engine is None:

                def _init_embed():
                    global _embed_engine
                    if _embed_engine is None:
                        logger.info(
                            "Loading shared Embedding Models (Bi-Encoder + Cross-Encoder)..."
                        )
                        _embed_engine = EmbeddingEngine()

                def _init_ner():
                    global _ner_engine
                    if _ner_engine is None:
                        logger.info(
                            "Loading shared GLiNER NER Model for cross-domain entity extraction..."
                        )
                        empty_df = pd.DataFrame(
                            columns=["Flavor Name", "Brand Name", "Aliases", "Is_Weak"]
                        )
                        _ner_engine = NEREngine(empty_df)

                with ThreadPoolExecutor(max_workers=2) as executor:
                    fut_embed = executor.submit(_init_embed)
                    fut_ner = executor.submit(_init_ner)
                    fut_embed.result()
                    fut_ner.result()

    return _embed_engine, _ner_engine


def _get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def get_pipeline(
    domain: str, check_for_updates: bool = False, force_sync: bool = False
) -> SKUMatcher:
    if domain not in (config.DOMAIN_MARKET, config.DOMAIN_FOOD):
        raise ValueError(f"Unknown domain '{domain}'. Must be 'market' or 'food'.")

    with _loader_lock:
        if domain in _pipelines and not force_sync and not check_for_updates:
            _model_statuses[domain]["pipeline"] = "ready"
            return _pipelines[domain]

    # Serialize concurrent builds of THIS domain only (market/food still build in parallel).
    with _pipeline_build_locks[domain]:
        # Re-check: another thread may have just finished building this domain while we
        # were waiting for the build lock, making our own build redundant.
        with _loader_lock:
            if domain in _pipelines and not force_sync and not check_for_updates:
                _model_statuses[domain]["pipeline"] = "ready"
                return _pipelines[domain]
            _model_statuses[domain]["pipeline"] = "loading"

        try:
            logger.info(f"[{domain.upper()}] Building pipeline...")
            sys.stdout.flush()

            embed_engine, ner_engine_shared = _get_shared_models()
            cat_df, brands_df = DataIngestion.load_catalog(
                config.GOOGLE_SHEET_ID, domain=domain, force_fetch=False
            )

            ner_engine = NEREngine(brands_df, domain=domain, shared_model=ner_engine_shared.model)
            logic_gates = LogicGates(embed_engine, brands_df=brands_df)
            cache_manager = CacheManager(ner_engine, embed_engine)

            classifier = get_classifier(domain, force_reset=force_sync)
            matcher = SKUMatcher(
                cat_df,
                brands_df,
                ner_engine,
                embed_engine,
                cache_manager,
                logic_gates,
                domain=domain,
                classifier=classifier,
                check_for_updates=check_for_updates,
                force_sync=force_sync,
            )

            with _loader_lock:
                _pipelines[domain] = matcher
                _domain_ner_engines[domain] = ner_engine
                _model_statuses[domain]["pipeline"] = "ready"
            logger.info(f"[{domain.upper()}] Pipeline ready.")
            sys.stdout.flush()
            return matcher
        except Exception as e:
            with _loader_lock:
                _model_statuses[domain]["pipeline"] = f"failed: {str(e)}"
            raise e


def get_classifier(domain: str, force_reset: bool = False) -> ZeroShotClassifier:
    if domain not in (config.DOMAIN_MARKET, config.DOMAIN_FOOD):
        raise ValueError(f"Unknown domain '{domain}'. Must be 'market' or 'food'.")

    with _loader_lock:
        if domain in _classifiers and not force_reset:
            _model_statuses[domain]["classifier"] = "ready"
            return _classifiers[domain]

    return _build_classifier(domain, force_reset)


def _build_classifier(domain: str, force_reset: bool) -> ZeroShotClassifier:
    # Serialize concurrent builds of THIS domain only (market/food still build in parallel).
    with _classifier_build_locks[domain]:
        # Re-check: another thread may have just finished training this domain's classifier
        # while we were waiting for the build lock, making our own build redundant.
        with _loader_lock:
            if domain in _classifiers and not force_reset:
                _model_statuses[domain]["classifier"] = "ready"
                return _classifiers[domain]
            _model_statuses[domain]["classifier"] = "training"

        return _train_classifier(domain, force_reset)


def _train_classifier(domain: str, force_reset: bool) -> ZeroShotClassifier:
    try:
        logger.info(f"[{domain.upper()}] Building classifier...")
        sys.stdout.flush()

        embed_engine, ner_engine_shared = _get_shared_models()
        vector_store = _get_vector_store()
        cat_df, brands_df = DataIngestion.load_catalog(
            config.GOOGLE_SHEET_ID, domain=domain, force_fetch=False
        )

        ner_engine = NEREngine(brands_df, domain=domain, shared_model=ner_engine_shared.model)
        cache_manager = CacheManager(ner_engine, embed_engine)
        dicts = DataIngestion.load_classifier_dictionaries(
            config.GOOGLE_SHEET_ID, domain=domain, force_fetch=False
        )

        def _build_bt_gk(df):
            bt_map = augment_bt_gk_map_with_training(df, {}, dicts.get("gk", []))
            umbrella = build_umbrella_from_training(df)
            third_tag_map = build_bt_third_tag_map_from_catalog(df, domain)
            return bt_map, umbrella, third_tag_map

        bt_gk_cache = cache_manager.get_or_build_bt_gk_cache(cat_df, domain, _build_bt_gk)
        bt_gk_map = bt_gk_cache["bt_gk_map"]
        umbrella = bt_gk_cache["umbrella"]
        third_tag_map = bt_gk_cache.get("third_tag_map", {})

        descriptions = build_descriptions(domain, dicts, bt_gk_map, third_tag_map)
        descriptions["bt_to_gk_umbrella"] = umbrella

        gk_tags = dicts.get("gk", [])
        if gk_tags:
            gk_embs = embed_engine.embed_dictionary_incremental(domain, "gk", gk_tags)
            vector_store.upsert_tags(
                gk_tags, gk_embs["dense"], gk_embs["sparse"], "gk", domain=domain, force=force_reset
            )

        bt_tags = dicts.get("bt", [])
        if bt_tags:
            bt_embs = embed_engine.embed_dictionary_incremental(domain, "bt", bt_tags)
            vector_store.upsert_tags(
                bt_tags, bt_embs["dense"], bt_embs["sparse"], "bt", domain=domain, force=force_reset
            )

        third_key = "region" if domain == config.DOMAIN_FOOD else "category"
        tt_tags = dicts.get(third_key, [])
        if tt_tags:
            tt_embs = embed_engine.embed_dictionary_incremental(domain, third_key, tt_tags)
            vector_store.upsert_tags(
                tt_tags,
                tt_embs["dense"],
                tt_embs["sparse"],
                third_key,
                domain=domain,
                force=force_reset,
            )

        classifier = ZeroShotClassifier(
            embed_engine, domain, descriptions, cat_df=cat_df, brands_df=brands_df
        )
        classifier.ner_engine = ner_engine

        with _loader_lock:
            _classifiers[domain] = classifier
            _domain_ner_engines[domain] = ner_engine
            _model_statuses[domain]["classifier"] = "ready"
        logger.info(f"[{domain.upper()}] Classifier ready.")
        sys.stdout.flush()
        return classifier
    except Exception as e:
        with _loader_lock:
            _model_statuses[domain]["classifier"] = f"failed: {str(e)}"
        raise e


def get_ner_engine(domain: str) -> NEREngine:
    """Returns a domain-specific NEREngine with the domain's dictionary loaded."""
    if domain not in (config.DOMAIN_MARKET, config.DOMAIN_FOOD):
        domain = config.DOMAIN_MARKET

    with _loader_lock:
        if domain in _domain_ner_engines:
            return _domain_ner_engines[domain]
        if (
            domain in _pipelines
            and hasattr(_pipelines[domain], "ner")
            and _pipelines[domain].ner is not None
        ):
            _domain_ner_engines[domain] = _pipelines[domain].ner
            return _domain_ner_engines[domain]
        if (
            domain in _classifiers
            and hasattr(_classifiers[domain], "ner_engine")
            and _classifiers[domain].ner_engine is not None
        ):
            _domain_ner_engines[domain] = _classifiers[domain].ner_engine
            return _domain_ner_engines[domain]

    # Build domain NEREngine reusing shared ONNX model
    _, ner_engine_shared = _get_shared_models()
    try:
        _, brands_df = DataIngestion.load_catalog(
            config.GOOGLE_SHEET_ID, domain=domain, force_fetch=False
        )
    except Exception as e:
        logger.warning(f"Could not load catalog for domain '{domain}' NER: {e}")
        brands_df = pd.DataFrame(columns=["Flavor Name", "Brand Name", "Aliases", "Is_Weak"])

    domain_ner = NEREngine(brands_df, domain=domain, shared_model=ner_engine_shared.model)
    with _loader_lock:
        _domain_ner_engines[domain] = domain_ner
    return domain_ner


def check_models_loaded(domain: str, task: str) -> None:
    """Helper to check if the required models are loaded for the requested domain and task."""
    needed = []
    if task == "matcher":
        needed.append(("pipeline", "pipeline model"))
    elif task == "classifier":
        needed.append(("classifier", "classifier model"))
    elif task == "pipeline":
        needed.append(("pipeline", "pipeline model"))
        needed.append(("classifier", "classifier model"))

    missing = []
    for model_key, label in needed:
        if _model_statuses.get(domain, {}).get(model_key) != "ready":
            missing.append(label)

    if missing:
        raise ValueError(
            f"Models for domain '{domain}' are not loaded (missing: {', '.join(missing)}). "
            f"Please call the /load-models endpoint to load the models first."
        )


def trigger_model_load(reload: bool = False) -> tuple[str, str]:
    """Thread-safe entrypoint for GetModelsStatus / ManageModels background warmup."""
    if not reload and are_all_models_ready() and not _model_load_lock.locked():
        return "success", "Models are already fully loaded."
    if not _model_load_lock.acquire(blocking=False):
        return "ignored", "Model loading is already in progress."
    try:
        threading.Thread(target=_bg_load_models, args=(reload,), daemon=True).start()
    except BaseException:
        _model_load_lock.release()
        raise
    return "started", "Model loading triggered."


def _fetch_snapshot_from_backend(domain: str) -> bool:
    """Fetches catalog snapshot (Feather files & dictionaries) from backend API and writes to CACHE_DIR."""
    import requests

    backend_url = getattr(config, "BACKEND_HTTP_URL", "http://127.0.0.1:8000").rstrip("/")
    api_key = getattr(config, "API_KEY", "bismillah")
    headers = {"X-API-Key": api_key}

    os.makedirs(config.CACHE_DIR, exist_ok=True)
    cat_path = os.path.join(config.CACHE_DIR, f"{domain}_catalog_mmap.feather")
    brands_path = os.path.join(config.CACHE_DIR, f"{domain}_brands_mmap.feather")
    dicts_path = os.path.join(config.CACHE_DIR, f"{domain}_classifier_dicts.json")

    try:
        # 1. Catalog Feather
        r_cat = requests.get(
            f"{backend_url}/catalog/{domain}/snapshot/catalog.feather",
            headers=headers,
            timeout=1.5,
        )
        if r_cat.status_code != 200:
            return False

        # 2. Brands Feather
        r_brands = requests.get(
            f"{backend_url}/catalog/{domain}/snapshot/brands.feather",
            headers=headers,
            timeout=1.5,
        )
        if r_brands.status_code != 200:
            return False

        with open(cat_path, "wb") as f:
            f.write(r_cat.content)
        with open(brands_path, "wb") as f:
            f.write(r_brands.content)

        # 3. Dictionaries JSON (optional best effort)
        try:
            r_dicts = requests.get(
                f"{backend_url}/catalog/{domain}/snapshot/dictionaries.json",
                headers=headers,
                timeout=1.5,
            )
            if r_dicts.status_code == 200:
                with open(dicts_path, "w", encoding="utf-8") as f:
                    f.write(r_dicts.text)
        except Exception as dict_err:
            logger.warning(f"[ENGINE] Failed to fetch dictionary snapshot for {domain}: {dict_err}")

        return True
    except Exception as e:
        logger.warning(f"[ENGINE] Failed to reach backend snapshot for {domain}: {e}")
        return False


def _rebuild_data_caches() -> None:
    """Engine owns data/cache: pull snapshot from backend; fallback to sheets/offline if backend unreachable."""
    DataIngestion.clear_mem_cache()
    for d in (config.DOMAIN_MARKET, config.DOMAIN_FOOD):
        try:
            ok = _fetch_snapshot_from_backend(d)
            if ok:
                logger.info(
                    f"[ENGINE] Successfully updated local cache from backend snapshot for {d}."
                )
            else:
                logger.warning(
                    f"[ENGINE] Backend snapshot unavailable for {d}; falling back to local/Sheets cache."
                )
                DataIngestion.load_catalog(config.GOOGLE_SHEET_ID, domain=d, force_fetch=False)
                DataIngestion.load_classifier_dictionaries(
                    config.GOOGLE_SHEET_ID, d, force_fetch=False
                )
        except Exception as e:
            # Keep serving from the previous on-disk cache rather than failing the reload.
            logger.error(f"[ENGINE] Cache rebuild failed for {d}, using existing cache: {e}")
            DataIngestion.clear_mem_cache(d)


def _bg_load_models(reload: bool = False) -> None:
    global _queue_worker
    try:
        logger.info("[ENGINE] Background model loading and warmup started...")
        if reload:
            reset_statuses(clear_caches=True)
            _rebuild_data_caches()
            gc.collect()

        try:
            embed_engine, ner_engine = _get_shared_models()
            if hasattr(embed_engine, "warmup"):
                embed_engine.warmup()
            if hasattr(ner_engine, "warmup"):
                ner_engine.warmup()
            gc.collect()
        except Exception as e:
            logger.warning(f"[ENGINE] Warmup failed: {e}")

        try:
            refresh_rules_cache()
            start_invalidation_listener()
            clear_flavor_cache()

            # Load domains sequentially with explicit gc to minimize peak memory allocation.
            # get_pipeline(d) builds and attaches get_classifier(d) internally.
            for d in (config.DOMAIN_MARKET, config.DOMAIN_FOOD):
                try:
                    get_pipeline(d)
                except Exception as e:
                    logger.error(f"[ENGINE] Failed to pre-load domain {d}: {e}")
                gc.collect()

            logger.info("[ENGINE] All ML models loaded and cached successfully.")
            from engine.pipeline.worker_runner import start_queue_worker

            # Only take queued jobs once models are ready; until then other workers can serve them
            if _queue_worker is None or not _queue_worker.is_alive():
                _queue_worker = start_queue_worker()
        except Exception as e:
            logger.error(f"[ENGINE] Background model loading failed: {e}")
            reset_statuses()
    finally:
        _model_load_lock.release()
