"""Distributed task queue, dead-letter replay, and operational status service."""

from typing import Any

from backend.core.exceptions import (
    ResourceConflictError,
    ResourceNotFoundError,
    ServiceUnavailableError,
)
from backend.repositories import (
    job_progress_repository,
    job_repository,
    sku_repository,
    task_queue_repository,
)
from backend.repositories.task_queue_repository import TaskQueueProtocol
from backend.services import rule_service


def get_active_queue() -> TaskQueueProtocol:
    """Returns the active Redis task queue or raises ServiceUnavailableError."""
    queue = task_queue_repository.get_task_queue()
    if queue is None:
        raise ServiceUnavailableError("Task queue is disabled (TASK_QUEUE_ENABLED).")
    return queue


def list_dead_letters(limit: int = 100) -> list[dict[str, Any]]:
    """Lists dead-lettered tasks from Redis."""
    queue = get_active_queue()
    try:
        return queue.fetch_dlq_items(limit=limit)
    except Exception as e:
        raise ServiceUnavailableError(f"Task queue unavailable: {e}") from e


def replay_dead_letter(task_id: str) -> dict[str, Any]:
    """Atomically resets SQL state and re-enqueues a dead-lettered task with compensation."""
    queue = get_active_queue()
    try:
        entry = next(
            (e for e in queue.fetch_dlq_items(limit=10000) if e["task_id"] == task_id),
            None,
        )
    except Exception as e:
        raise ServiceUnavailableError(f"Task queue unavailable: {e}") from e

    if entry is None:
        raise ResourceNotFoundError("Dead-letter entry not found")

    job_id = entry["job_id"]
    payload = entry.get("payload") or {}
    batch_id = entry.get("batch_id") or payload.get("batch_id") or job_id

    if not job_repository.requeue_job_for_replay(job_id, batch_id):
        raise ResourceNotFoundError("Job record not found")

    try:
        new_task_id = queue.replay_dlq_task(task_id)
    except Exception as e:
        try:
            job_repository.mark_job_failed_after_replay_error(
                job_id, batch_id, "Replay queue failed", include_batch=True
            )
        except Exception:
            pass
        raise ServiceUnavailableError(f"Task queue replay failed: {e}") from e

    if new_task_id is None:
        try:
            job_repository.mark_job_failed_after_replay_error(
                job_id, batch_id, "Concurrent replay detected", include_batch=False
            )
        except Exception:
            pass
        raise ResourceConflictError("Dead-letter entry was already replayed")

    job_progress_repository.clear_job_progress(job_id)
    job_progress_repository.set_job_progress(job_id, pct=0.0, stage="queued")
    job_progress_repository.publish_job_event(
        {"id": job_id, "status": "queued", "current_stage": "queued", "progress_pct": 0.0}
    )
    return {"message": "Replay queued", "job_id": job_id, "task_id": new_task_id}


def fetch_system_status() -> dict[str, Any]:
    """Returns operational counters for the portal status strip."""
    processed = sku_repository.count_all_processed_skus()

    queue_depth, dlq_size, queue_online = None, None, False
    try:
        queue = task_queue_repository.get_task_queue()
        if queue is not None:
            dlq_size = queue.count_dlq_items()
            queue_depth = queue.get_queue_stats()
            queue_online = True
    except Exception:
        pass

    return {
        "processed_skus": processed,
        "rule_version": rule_service.get_rule_version(),
        "queue_online": queue_online,
        "queue_depth": queue_depth,
        "dlq_size": dlq_size,
    }
