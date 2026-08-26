"""
SKU MatchOps Engine - Standalone Inference & Processing Microservice
Runs on port 8001 (or ENGINE_PORT) to handle compute-intensive ML workloads.
"""

import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engine import config
from engine.audit_engine import run_sku_audit
from engine.data_pipeline.vector_store import VectorStore
from engine.processor import process_request
from engine.resource_loader import (
    _get_shared_models,
    _model_statuses,
    get_classifier,
    get_pipeline,
    reset_statuses,
)
from engine.rules_engine import refresh_rules_cache
from engine.template_suggest import suggest_tags_from_template
from engine.worker_runner import cancel_job, enqueue_batch_job
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# Configure logging
logger = logging.getLogger("matchops.engine.server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

app = FastAPI(
    title="SKU MatchOps ML Engine",
    description="Dedicated ML Inference Microservice for SKU Matching and Classification.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model_load_lock = threading.Lock()
_loading_in_progress = False


# --- Request Schemas ---

class SKUItemPayload(BaseModel):
    name: str
    price: Optional[float] = 0.0
    description: Optional[str] = ""
    category: Optional[str] = ""


class BatchProcessRequest(BaseModel):
    job_id: str
    task: str = "pipeline"
    domain: str = "market"
    skus: List[SKUItemPayload]
    backend_url: str = "http://backend:8000"
    callback_url: Optional[str] = None
    sheet_name: Optional[str] = None
    spreadsheet_id: Optional[str] = None


class RunSingleRequest(BaseModel):
    sku_name: str
    domain: str = "market"
    task: str = "pipeline"
    price: Optional[float] = 0.0
    description: Optional[str] = ""
    category: Optional[str] = ""


class SuggestRequest(BaseModel):
    sku_name: str
    domain: str = "market"
    current_bt: Optional[str] = ""
    exclude_bt: Optional[str] = ""
    exclude_gk: Optional[str] = ""


class VectorSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    score_threshold: Optional[float] = None
    filters: Optional[Dict[str, Any]] = None


# --- Background Model Pre-loader ---

def _bg_load_models():
    global _loading_in_progress
    logger.info("[ENGINE] Background model loading and warmup started...")
    try:
        for d in _model_statuses:
            _model_statuses[d]["pipeline"] = "loading"
            _model_statuses[d]["classifier"] = "training"

        embed_engine, ner_engine = _get_shared_models()
        if hasattr(embed_engine, 'warmup'):
            embed_engine.warmup()
        if hasattr(ner_engine, 'warmup'):
            ner_engine.warmup()
    except Exception as e:
        logger.warning(f"[ENGINE] Warmup failed: {e}")

    try:
        refresh_rules_cache()

        def _init_domain(d):
            try:
                get_pipeline(d)
                get_classifier(d)
            except Exception as e:
                logger.error(f"[ENGINE] Failed to pre-load domain {d}: {e}")

        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(_init_domain, [config.DOMAIN_MARKET, config.DOMAIN_FOOD]))

        logger.info("[ENGINE] All ML models loaded and cached successfully.")
    except Exception as e:
        logger.error(f"[ENGINE] Background model loading failed: {e}")
        reset_statuses()
    finally:
        _loading_in_progress = False


# --- Endpoints ---

@app.get("/health")
def health():
    market_ready = _model_statuses["market"]["pipeline"] == "ready" and _model_statuses["market"]["classifier"] == "ready"
    food_ready = _model_statuses["food"]["pipeline"] == "ready" and _model_statuses["food"]["classifier"] == "ready"
    return {
        "status": "ok",
        "service": "matchops-engine",
        "loaded_domains": [d for d in _model_statuses if _model_statuses[d]["pipeline"] == "ready"],
        "all_models_ready": market_ready and food_ready,
        "model_statuses": _model_statuses,
        "loading_in_progress": _loading_in_progress
    }


@app.get("/models/status")
@app.get("/models-status")
def models_status():
    market_ready = _model_statuses["market"]["pipeline"] == "ready" and _model_statuses["market"]["classifier"] == "ready"
    food_ready = _model_statuses["food"]["pipeline"] == "ready" and _model_statuses["food"]["classifier"] == "ready"
    is_ready = market_ready and food_ready and not _loading_in_progress
    return {
        "status": "ready" if is_ready else ("loading" if _loading_in_progress else "unloaded"),
        "loaded": is_ready,
        "details": _model_statuses,
        "loading_in_progress": _loading_in_progress
    }


@app.post("/load-models")
def load_models(background_tasks: BackgroundTasks):
    global _loading_in_progress
    if _loading_in_progress:
        return {"status": "ignored", "message": "Model loading is already in progress."}

    market_ready = _model_statuses["market"]["pipeline"] == "ready" and _model_statuses["market"]["classifier"] == "ready"
    food_ready = _model_statuses["food"]["pipeline"] == "ready" and _model_statuses["food"]["classifier"] == "ready"
    if market_ready and food_ready:
        return {"status": "success", "message": "Models are already fully loaded."}

    acquired = _model_load_lock.acquire(blocking=False)
    if not acquired:
        return {"status": "ignored", "message": "Model loading is already in progress."}

    _loading_in_progress = True
    for d in _model_statuses:
        _model_statuses[d]["pipeline"] = "loading"
        _model_statuses[d]["classifier"] = "training"

    def run_with_lock():
        try:
            _bg_load_models()
        finally:
            _model_load_lock.release()

    background_tasks.add_task(run_with_lock)
    return {"status": "started", "message": "Model loading initiated in background."}


@app.post("/engine/process-batch")
def process_batch(request: BatchProcessRequest):
    """Enqueues a batch job for background processing by the engine worker."""
    if not request.skus:
        raise HTTPException(status_code=400, detail="Empty SKU list provided.")

    payload = {
        "job_id": request.job_id,
        "task": request.task,
        "domain": request.domain,
        "skus": [s.model_dump() for s in request.skus],
        "backend_url": request.backend_url,
        "callback_url": request.callback_url,
        "sheet_name": request.sheet_name,
        "spreadsheet_id": request.spreadsheet_id,
    }

    result = enqueue_batch_job(payload)
    return result


@app.post("/engine/run-single")
def run_single(request: RunSingleRequest):
    """Executes synchronous single-SKU inference for interactive playground."""
    try:
        sku_dict = {
            "name": request.sku_name,
            "price": request.price or 0.0,
            "description": request.description or "",
            "category": request.category or ""
        }
        res = process_request(
            skus=[sku_dict],
            task=request.task,
            domain=request.domain
        )
        return res
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"[ENGINE] Single run failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/engine/suggest")
def suggest(request: SuggestRequest):
    """Executes template-aware tag suggestion."""
    try:
        res = suggest_tags_from_template(
            sku_name=request.sku_name,
            domain=request.domain,
            current_bt=request.current_bt,
            exclude_bt=request.exclude_bt,
            exclude_gk=request.exclude_gk
        )
        return res
    except Exception as e:
        logger.error(f"[ENGINE] Template suggest failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/engine/jobs/{job_id}/cancel")
def cancel_job_endpoint(job_id: str):
    """Instructs the engine worker to abort processing for the specified job."""
    cancel_job(job_id)
    return {"status": "cancelled", "job_id": job_id}


@app.post("/engine/reload-models")
def reload_models(background_tasks: BackgroundTasks):
    """Forces cache clearing and model reload."""
    reset_statuses()
    return load_models(background_tasks)


# --- Vector Database & Audit Endpoints ---

@app.get("/engine/vector-db/collections")
def engine_list_collections():
    """Lists vector database collections from Qdrant."""
    try:
        client = VectorStore.get_client()
        collections = client.get_collections()
        return {"collections": [c.name for c in collections.collections]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/engine/vector-db/collections/{name}")
def engine_get_collection(name: str):
    """Gets detailed info for a Qdrant vector database collection."""
    try:
        client = VectorStore.get_client()
        info = client.get_collection(collection_name=name)
        return info.dict()
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Collection not found: {e}")


@app.post("/engine/vector-db/collections/{name}/search")
def engine_search_collection(name: str, request: VectorSearchRequest):
    """Executes vector similarity search on Qdrant using BGE-M3 embeddings."""
    try:
        client = VectorStore.get_client()
        embed_engine, _ = _get_shared_models()
        encoded = embed_engine.encode([request.query])
        query_vector = encoded["dense"][0]
        
        qdrant_filter = None
        if request.filters:
            conditions = []
            for key, val in request.filters.items():
                conditions.append(
                    qmodels.FieldCondition(
                        key=key,
                        match=qmodels.MatchValue(value=val)
                    )
                )
            if conditions:
                qdrant_filter = qmodels.Filter(must=conditions)
        
        search_result = client.query_points(
            collection_name=name,
            query=query_vector.tolist() if hasattr(query_vector, 'tolist') else query_vector,
            using="dense",
            limit=request.top_k,
            query_filter=qdrant_filter,
            score_threshold=request.score_threshold
        ).points
        
        return {
            "results": [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload
                }
                for hit in search_result
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/engine/audit")
def engine_audit(request: RunSingleRequest):
    """Executes a transparent diagnostic SKU audit."""
    try:
        res = run_sku_audit(
            sku_name=request.sku_name,
            domain=request.domain,
            task=request.task,
            price=request.price or 0.0,
            description=request.description or "",
            category=request.category or ""
        )
        return res
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"[ENGINE] Audit run failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("ENGINE_PORT", 8001))
    host = os.getenv("ENGINE_HOST", "0.0.0.0")
    logger.info(f"Starting SKU MatchOps Engine on {host}:{port}...")
    uvicorn.run("engine.server:app", host=host, port=port, reload=False)
