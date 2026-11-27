"""System startup/shutdown lifecycle, and health/readiness probes."""

import logging
from typing import Any

from backend.repositories import (
    database_init_repository,
    job_progress_repository,
    job_repository,
    task_queue_repository,
)
from backend.services import engine_gateway_service, rule_service

logger = logging.getLogger("matchops.services.system")


def run_startup_sequence() -> None:
    """Initializes PostgreSQL schema, syncs rules, verifies Redis, and reconciles stalled jobs."""
    try:
        database_init_repository.initialize_database_schema()
        database_init_repository.verify_required_tables()
        logger.info("Database schemas and tables verified.")
    except Exception as e:
        logger.error(f"FATAL: Failed to initialize/verify database tables on startup: {e}")
        raise

    try:
        rule_service.sync_rules_everywhere()
    except Exception as e:
        logger.warning(f"Failed to sync rules on startup: {e}")

    queue = task_queue_repository.get_task_queue()
    if queue is None or not queue.ping_broker():
        raise RuntimeError(
            "FATAL: Redis is required and unreachable at startup. Check REDIS_URL and redis container."
        )
    live_job_ids = list(queue.list_tracked_job_ids())

    if live_job_ids:
        try:
            stale_ids = job_repository.fetch_stale_live_job_ids(live_job_ids)
            for jid in stale_ids:
                queue.request_cancel(jid)
            if stale_ids:
                live_job_ids = [jid for jid in live_job_ids if str(jid) not in stale_ids]
                logger.info(
                    f"Purged {len(stale_ids)} terminal/cancelled job(s) from Redis queue on startup."
                )
        except Exception as e:
            logger.warning(f"Failed to purge stale Redis jobs on startup: {e}")

    try:
        stuck_count = job_repository.fail_orphaned_jobs_and_batches(live_job_ids)
        if stuck_count:
            logger.warning(
                f"Marked {stuck_count} job(s) left in queued/running state as failed after restart."
            )
    except Exception as e:
        logger.error(f"Failed to reconcile stuck jobs on startup: {e}")


async def close_async_resources() -> None:
    """Releases asynchronous Redis connection pools on gateway shutdown."""
    await job_progress_repository.close_async_redis_pool()


def check_system_health() -> dict[str, Any]:
    """Returns basic liveness status alongside ML Engine domain readiness."""
    engine_status = engine_gateway_service.get_engine_health()
    return {
        "status": "ok",
        "service": "matchops-backend",
        "loaded_domains": engine_status.get("loaded_domains", []),
        "engine": engine_status,
    }


def check_system_readiness() -> tuple[bool, dict[str, Any]]:
    """Checks PostgreSQL, Redis, and ML Engine model readiness."""
    db_ok = database_init_repository.check_database_health()
    redis_ok = task_queue_repository.ping_redis()
    engine_status = engine_gateway_service.get_engine_health()
    engine_ok = bool(engine_status.get("all_models_ready"))
    is_ready = db_ok and redis_ok and engine_ok
    payload = {
        "status": "ready" if is_ready else "not_ready",
        "db": db_ok,
        "redis": redis_ok,
        "engine": engine_status,
    }
    return is_ready, payload
