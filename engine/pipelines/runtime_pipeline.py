"""
SKU MatchOps Engine - Model & Resource Loader
Handles loading, warm-up, and singleton caching of ML pipelines, NER, and Classifiers.
"""

import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from engine.core import config
from engine.core.onnx_factory import trim_process_memory
from engine.inference.embedding_model import EmbeddingModel
from engine.inference.gliner_model import GlinerModel
from engine.inference.reranker_model import RerankerModel
from engine.pipelines.classification_pipeline import TaxonomyClassifier
from engine.pipelines.indexing_pipeline import CacheManager
from engine.pipelines.ingestion_pipeline import DataIngestion
from engine.pipelines.matching_logic_pipeline import LogicGates
from engine.pipelines.matching_pipeline import SKUMatcher
from engine.pipelines.rules_evaluation_pipeline import clear_flavor_cache
from engine.pipelines.rules_loader_pipeline import (
    refresh_rules_cache,
    start_invalidation_listener,
)
from engine.preprocessing.taxonomy_preprocessor import (
    augment_bt_gk_map,
    build_bt_third_tag_map,
    build_taxonomy_descriptions,
    build_umbrella_gk_map,
)
from engine.vector_store.vector_store import VectorStore

logger = logging.getLogger("matchops.engine.loader")

_pipelines: dict[str, SKUMatcher] = {}
_classifiers: dict[str, TaxonomyClassifier] = {}
_domain_ner_engines: dict[str, GlinerModel] = {}
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


def get_model_statuses() -> dict[str, dict[str, str]]:
    """Returns per-domain pipeline and classifier readiness statuses."""
    return _model_statuses


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


_embed_engine = None
_rerank_engine = None
_ner_engine = None


def _get_shared_models():
    """Loads the embedding, reranker, and NER ONNX models concurrently."""
    global _embed_engine, _rerank_engine, _ner_engine
    if _embed_engine is None or _rerank_engine is None or _ner_engine is None:
        with _loader_lock:
            if _embed_engine is None or _rerank_engine is None or _ner_engine is None:

                def _init_embed():
                    global _embed_engine
                    if _embed_engine is None:
                        logger.info("Loading shared Embedding ONNX model...")
                        _embed_engine = EmbeddingModel()

                def _init_rerank():
                    global _rerank_engine
                    if _rerank_engine is None:
                        logger.info("Loading shared Reranker ONNX model...")
                        _rerank_engine = RerankerModel()

                def _init_ner():
                    global _ner_engine
                    if _ner_engine is None:
                        logger.info(
                            "Loading shared GLiNER NER Model for cross-domain entity extraction..."
                        )
                        empty_df = pd.DataFrame(
                            columns=["Flavor Name", "Brand Name", "Aliases", "Is_Weak"]
                        )
                        _ner_engine = GlinerModel(empty_df)

                with ThreadPoolExecutor(max_workers=3) as executor:
                    fut_embed = executor.submit(_init_embed)
                    fut_rerank = executor.submit(_init_rerank)
                    fut_ner = executor.submit(_init_ner)
                    fut_embed.result()
                    fut_rerank.result()
                    fut_ner.result()

    return _embed_engine, _ner_engine


def _get_reranker_model(embed_engine=None):
    """Returns the active RerankerModel (or compatible mock/eval composite) without extra disk I/O."""
    if _rerank_engine is not None and (
        embed_engine is None or isinstance(embed_engine, EmbeddingModel)
    ):
        return _rerank_engine
    if embed_engine is not None and not isinstance(embed_engine, EmbeddingModel):
        if getattr(embed_engine, "cross_session", None) is not None:
            return embed_engine
        return None
    _get_shared_models()
    return _rerank_engine


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

    with _pipeline_build_locks[domain]:
        with _loader_lock:
            if domain in _pipelines and not force_sync and not check_for_updates:
                _model_statuses[domain]["pipeline"] = "ready"
                return _pipelines[domain]
            _model_statuses[domain]["pipeline"] = "loading"

        try:
            logger.info(f"[{domain.upper()}] Building pipeline...")
            sys.stdout.flush()

            embed_engine, ner_engine_shared = _get_shared_models()
            rerank_engine = _get_reranker_model(embed_engine)
            cat_df, brands_df = DataIngestion.load_catalog(
                config.GOOGLE_SHEET_ID, domain=domain, force_fetch=False
            )

            classifier = get_classifier(domain, force_reset=force_sync)
            ner_engine = _get_or_create_domain_ner(
                domain, brands_df, ner_engine_shared.model, force_refresh=False
            )
            logic_gates = LogicGates(embed_engine, brands_df=brands_df)
            cache_manager = CacheManager(ner_engine, embed_engine)

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
                reranker=rerank_engine,
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


def get_classifier(domain: str, force_reset: bool = False) -> TaxonomyClassifier:
    if domain not in (config.DOMAIN_MARKET, config.DOMAIN_FOOD):
        raise ValueError(f"Unknown domain '{domain}'. Must be 'market' or 'food'.")

    with _loader_lock:
        if domain in _classifiers and not force_reset:
            _model_statuses[domain]["classifier"] = "ready"
            return _classifiers[domain]

    return _build_classifier(domain, force_reset)


def _build_classifier(domain: str, force_reset: bool) -> TaxonomyClassifier:
    with _classifier_build_locks[domain]:
        with _loader_lock:
            if domain in _classifiers and not force_reset:
                _model_statuses[domain]["classifier"] = "ready"
                return _classifiers[domain]
            _model_statuses[domain]["classifier"] = "training"

        return _train_classifier(domain, force_reset)


def _train_classifier(domain: str, force_reset: bool) -> TaxonomyClassifier:
    try:
        logger.info(f"[{domain.upper()}] Building classifier...")
        sys.stdout.flush()

        embed_engine, ner_engine_shared = _get_shared_models()
        vector_store = _get_vector_store()
        cat_df, brands_df = DataIngestion.load_catalog(
            config.GOOGLE_SHEET_ID, domain=domain, force_fetch=False
        )

        ner_engine = _get_or_create_domain_ner(
            domain, brands_df, ner_engine_shared.model, force_refresh=force_reset
        )
        cache_manager = CacheManager(ner_engine, embed_engine)
        dicts = DataIngestion.load_classifier_dictionaries(
            config.GOOGLE_SHEET_ID, domain=domain, force_fetch=False
        )

        def _build_bt_gk(df):
            bt_map = augment_bt_gk_map(df, {}, dicts.get("gk", []))
            umbrella = build_umbrella_gk_map(df)
            third_tag_map = build_bt_third_tag_map(df, domain)
            return bt_map, umbrella, third_tag_map

        bt_gk_cache = cache_manager.get_or_build_bt_gk_cache(cat_df, domain, _build_bt_gk)
        bt_gk_map = bt_gk_cache["bt_gk_map"]
        umbrella = bt_gk_cache["umbrella"]
        third_tag_map = bt_gk_cache.get("third_tag_map", {})

        descriptions = build_taxonomy_descriptions(domain, dicts, bt_gk_map, third_tag_map)
        descriptions["bt_to_gk_umbrella"] = umbrella

        for tag_key, dict_type in (
            ("gk", "gk"),
            ("bt", "bt"),
            (
                "region" if domain == config.DOMAIN_FOOD else "category",
                "region" if domain == config.DOMAIN_FOOD else "category",
            ),
        ):
            tag_list = dicts.get(tag_key, [])
            if tag_list:
                if (
                    not force_reset
                    and hasattr(vector_store, "check_tags_synced")
                    and vector_store.check_tags_synced(domain, dict_type, len(set(tag_list)))
                ):
                    continue
                tag_embs = embed_engine.embed_dictionary_incremental(domain, dict_type, tag_list)
                vector_store.upsert_tag_vectors(
                    tag_list,
                    tag_embs["dense"],
                    sparse_embs=tag_embs.get("sparse"),
                    dict_type=dict_type,
                    domain=domain,
                    force=force_reset,
                    bm25_embs=tag_embs.get("bm25"),
                )
                del tag_embs

        classifier = TaxonomyClassifier(
            embed_engine,
            domain,
            descriptions,
            cat_df=cat_df,
            brands_df=brands_df,
            reranker=_get_reranker_model(embed_engine),
        )
        classifier.ner_engine = ner_engine
        if getattr(classifier, "cold_start_router", None) is not None and hasattr(
            classifier.cold_start_router, "registry"
        ):
            try:
                classifier.cold_start_router.registry.ensure_zero_shot_embeddings(embed_engine)
            except Exception as zs_err:
                logger.debug(f"[{domain.upper()}] Deferred zero-shot embedding warmup: {zs_err}")

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


def _get_or_create_domain_ner(
    domain: str,
    brands_df: pd.DataFrame | None = None,
    shared_model=None,
    force_refresh: bool = False,
) -> GlinerModel:
    """Reuse or build a domain-scoped GlinerModel sharing the base ONNX model."""
    with _loader_lock:
        if not force_refresh:
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

    if shared_model is None:
        _, ner_engine_shared = _get_shared_models()
        shared_model = ner_engine_shared.model
    if brands_df is None:
        try:
            _, brands_df = DataIngestion.load_catalog(
                config.GOOGLE_SHEET_ID, domain=domain, force_fetch=False
            )
        except Exception as e:
            logger.warning(f"Could not load catalog for domain '{domain}' NER: {e}")
            brands_df = pd.DataFrame(columns=["Flavor Name", "Brand Name", "Aliases", "Is_Weak"])

    domain_ner = GlinerModel(brands_df, domain=domain, shared_model=shared_model)
    with _loader_lock:
        if not force_refresh and domain in _domain_ner_engines:
            return _domain_ner_engines[domain]
        _domain_ner_engines[domain] = domain_ner
    return domain_ner


def get_ner_engine(domain: str) -> GlinerModel:
    """Returns a domain-specific GlinerModel with the domain's dictionary loaded."""
    if domain not in (config.DOMAIN_MARKET, config.DOMAIN_FOOD):
        domain = config.DOMAIN_MARKET
    return _get_or_create_domain_ner(domain)


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
            f"Please call the ManageModels gRPC RPC to load the models first."
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

    cache_dir = Path(config.CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cat_path = cache_dir / f"{domain}_catalog_mmap.feather"
    brands_path = cache_dir / f"{domain}_brands_mmap.feather"
    dicts_path = cache_dir / f"{domain}_classifier_dicts.json"

    try:
        r_cat = requests.get(
            f"{backend_url}/catalog/{domain}/snapshot/catalog.feather",
            headers=headers,
            timeout=1.5,
        )
        if r_cat.status_code != 200:
            return False

        r_brands = requests.get(
            f"{backend_url}/catalog/{domain}/snapshot/brands.feather",
            headers=headers,
            timeout=1.5,
        )
        if r_brands.status_code != 200:
            return False

        cat_path.write_bytes(r_cat.content)
        brands_path.write_bytes(r_brands.content)

        try:
            r_dicts = requests.get(
                f"{backend_url}/catalog/{domain}/snapshot/dictionaries.json",
                headers=headers,
                timeout=1.5,
            )
            if r_dicts.status_code == 200:
                dicts_path.write_text(r_dicts.text, encoding="utf-8")
        except Exception as dict_err:
            logger.warning(f"[ENGINE] Failed to fetch dictionary snapshot for {domain}: {dict_err}")

        return True
    except Exception as e:
        logger.warning(f"[ENGINE] Failed to reach backend snapshot for {domain}: {e}")
        return False


def _rebuild_data_caches() -> None:
    """Pull snapshot from backend; fallback to local/Sheets cache if backend unreachable."""
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
            logger.error(f"[ENGINE] Cache rebuild failed for {d}, using existing cache: {e}")
            DataIngestion.clear_mem_cache(d)


def _bg_load_models(reload: bool = False) -> None:
    global _queue_worker
    try:
        logger.info("[ENGINE] Background model loading and warmup started...")
        if reload:
            reset_statuses(clear_caches=True)
            _rebuild_data_caches()
            trim_process_memory()

        try:
            embed_engine, ner_engine = _get_shared_models()
            rerank_engine = _get_reranker_model(embed_engine)
            if hasattr(embed_engine, "warmup"):
                embed_engine.warmup()
            if hasattr(rerank_engine, "warmup"):
                rerank_engine.warmup()
            if hasattr(ner_engine, "warmup"):
                ner_engine.warmup()
            trim_process_memory()
        except Exception as e:
            logger.warning(f"[ENGINE] Warmup failed: {e}")

        try:
            refresh_rules_cache()
            start_invalidation_listener()
            clear_flavor_cache()

            for d in (config.DOMAIN_MARKET, config.DOMAIN_FOOD):
                try:
                    get_pipeline(d)
                except Exception as e:
                    logger.error(f"[ENGINE] Failed to pre-load domain {d}: {e}")
                trim_process_memory()

            logger.info("[ENGINE] All ML models loaded and cached successfully.")
            from engine.pipelines.worker_pipeline import start_queue_worker

            _queue_worker = start_queue_worker()
        except Exception as e:
            logger.error(f"[ENGINE] Background model loading failed: {e}")
            reset_statuses()
    finally:
        _model_load_lock.release()
