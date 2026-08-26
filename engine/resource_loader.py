"""
SKU MatchOps Engine - Model & Resource Loader
Handles loading, warm-up, and singleton caching of ML pipelines, NER, and Classifiers.
"""

from concurrent.futures import ThreadPoolExecutor
import logging
import sys
import threading
import pandas as pd
from engine import config
from engine.classification.classifier import ZeroShotClassifier
from engine.classification.loader import (
    augment_bt_gk_map_with_training,
    build_descriptions,
    build_umbrella_from_training,
    build_bt_third_tag_map_from_catalog,
)
from engine.data_pipeline.cache_manager import CacheManager
from engine.data_pipeline.ingestion import DataIngestion
from engine.data_pipeline.vector_store import VectorStore
from engine.matching.matcher import SKUMatcher
from engine.matching.logic_gates import LogicGates
from engine.nlp.embedding_engine import EmbeddingEngine
from engine.nlp.ner_engine import NEREngine

logger = logging.getLogger("matchops.engine.loader")

# Lazy pipeline cache
_pipelines: dict[str, SKUMatcher] = {}
_classifiers: dict[str, ZeroShotClassifier] = {}
_vector_store = None
_loader_lock = threading.RLock()

_model_statuses = {
    "market": {"pipeline": "idle", "classifier": "idle"},
    "food": {"pipeline": "idle", "classifier": "idle"}
}

def reset_statuses():
    for domain in _model_statuses:
        _model_statuses[domain]["pipeline"] = "idle"
        _model_statuses[domain]["classifier"] = "idle"

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
                        logger.info("Loading shared Embedding Models (Bi-Encoder + Cross-Encoder)...")
                        _embed_engine = EmbeddingEngine()

                def _init_ner():
                    global _ner_engine
                    if _ner_engine is None:
                        logger.info("Loading shared GLiNER NER Model for cross-domain entity extraction...")
                        empty_df = pd.DataFrame(columns=["Flavor Name", "Brand Name", "Aliases", "Is_Weak"])
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

def get_pipeline(domain: str) -> SKUMatcher:
    if domain not in (config.DOMAIN_MARKET, config.DOMAIN_FOOD):
        raise ValueError(f"Unknown domain '{domain}'. Must be 'market' or 'food'.")

    with _loader_lock:
        if domain in _pipelines:
            _model_statuses[domain]["pipeline"] = "ready"
            return _pipelines[domain]
        _model_statuses[domain]["pipeline"] = "loading"

    try:
        logger.info(f"[{domain.upper()}] Building pipeline...")
        sys.stdout.flush()

        embed_engine, ner_engine_shared = _get_shared_models()
        cat_df, brands_df = DataIngestion.load_catalog(
            config.GOOGLE_SHEET_ID, domain=domain
        )

        ner_engine = NEREngine(brands_df, domain=domain, shared_model=ner_engine_shared.model)
        logic_gates = LogicGates(embed_engine, brands_df=brands_df)
        cache_manager = CacheManager(ner_engine, embed_engine)

        classifier = get_classifier(domain)
        matcher = SKUMatcher(
            cat_df, brands_df, ner_engine, embed_engine,
            cache_manager, logic_gates, domain=domain,
            classifier=classifier
        )

        with _loader_lock:
            _pipelines[domain] = matcher
            _model_statuses[domain]["pipeline"] = "ready"
        logger.info(f"[{domain.upper()}] Pipeline ready.")
        sys.stdout.flush()
        return matcher
    except Exception as e:
        with _loader_lock:
            _model_statuses[domain]["pipeline"] = f"failed: {str(e)}"
        raise e

def get_classifier(domain: str) -> ZeroShotClassifier:
    if domain not in (config.DOMAIN_MARKET, config.DOMAIN_FOOD):
        raise ValueError(f"Unknown domain '{domain}'. Must be 'market' or 'food'.")

    with _loader_lock:
        if domain in _classifiers:
            _model_statuses[domain]["classifier"] = "ready"
            return _classifiers[domain]
        _model_statuses[domain]["classifier"] = "training"

    try:
        logger.info(f"[{domain.upper()}] Building classifier...")
        sys.stdout.flush()

        embed_engine, ner_engine_shared = _get_shared_models()
        vector_store = _get_vector_store()
        cat_df, brands_df = DataIngestion.load_catalog(config.GOOGLE_SHEET_ID, domain=domain)
        
        ner_engine = NEREngine(brands_df, domain=domain, shared_model=ner_engine_shared.model)
        cache_manager = CacheManager(ner_engine, embed_engine)
        dicts = DataIngestion.load_classifier_dictionaries(config.GOOGLE_SHEET_ID, domain=domain)

        def _build_bt_gk(df):
            bt_map = augment_bt_gk_map_with_training(df, {}, dicts.get("gk", []))
            umbrella = build_umbrella_from_training(df)
            third_tag_map = build_bt_third_tag_map_from_catalog(df, domain)
            return bt_map, umbrella, third_tag_map

        bt_gk_cache = cache_manager.get_or_build_bt_gk_cache(cat_df, domain, _build_bt_gk)
        bt_gk_map   = bt_gk_cache["bt_gk_map"]
        umbrella    = bt_gk_cache["umbrella"]
        third_tag_map = bt_gk_cache.get("third_tag_map", {})

        descriptions = build_descriptions(domain, dicts, bt_gk_map, third_tag_map)
        descriptions["bt_to_gk_umbrella"] = umbrella

        gk_tags = dicts.get("gk", [])
        if gk_tags:
            gk_embs = embed_engine.embed_dictionary_incremental(domain, "gk", gk_tags)
            vector_store.upsert_tags(gk_tags, gk_embs["dense"], gk_embs["sparse"], "gk", domain=domain)

        bt_tags = dicts.get("bt", [])
        if bt_tags:
            bt_embs = embed_engine.embed_dictionary_incremental(domain, "bt", bt_tags)
            vector_store.upsert_tags(bt_tags, bt_embs["dense"], bt_embs["sparse"], "bt", domain=domain)

        third_key = "region" if domain == config.DOMAIN_FOOD else "category"
        tt_tags = dicts.get(third_key, [])
        if tt_tags:
            tt_embs = embed_engine.embed_dictionary_incremental(domain, third_key, tt_tags)
            vector_store.upsert_tags(tt_tags, tt_embs["dense"], tt_embs["sparse"], third_key, domain=domain)

        classifier = ZeroShotClassifier(embed_engine, domain, descriptions, cat_df=cat_df, brands_df=brands_df)
        classifier.ner_engine = ner_engine

        with _loader_lock:
            _classifiers[domain] = classifier
            _model_statuses[domain]["classifier"] = "ready"
        logger.info(f"[{domain.upper()}] Classifier ready.")
        sys.stdout.flush()
        return classifier
    except Exception as e:
        with _loader_lock:
            _model_statuses[domain]["classifier"] = f"failed: {str(e)}"
        raise e


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
