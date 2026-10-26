from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text

from backend.app.api.endpoints.engine_callbacks import _job_eta, _job_progress
from backend.app.core.db import get_engine
from backend.app.core.security import Principal, require_admin
from engine.pipeline.task_queue import TaskQueue, get_task_queue

router = APIRouter()


def _queue() -> TaskQueue:
    queue = get_task_queue()
    if queue is None:
        raise HTTPException(status_code=503, detail="Task queue is disabled (TASK_QUEUE_ENABLED).")
    return queue


def _tenant_items(queue: TaskQueue, principal: Principal) -> list[dict]:
    """DLQ entries of the caller's tenant (jobs queued before tenancy count as 'default')."""
    return [
        e
        for e in queue.dlq_items(limit=10000)
        if e["payload"].get("tenant_id", "default") == principal.tenant_id
    ]


@router.get("/queue/dlq")
def list_dead_letters(limit: int = 100, principal: Principal = Depends(require_admin)):
    """The tenant's dead-lettered tasks: payload plus job_id, batch_id, error, attempts, failed_at."""
    try:
        return _tenant_items(_queue(), principal)[:limit]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Task queue unavailable: {e}")


@router.post("/queue/dlq/{task_id}/replay")
def replay_dead_letter(task_id: str, principal: Principal = Depends(require_admin)):
    """Re-queues a dead-lettered task with a fresh retry budget and marks its job queued again."""
    queue = _queue()
    entry = next((e for e in _tenant_items(queue, principal) if e["task_id"] == task_id), None)
    if entry is None:
        raise HTTPException(status_code=404, detail="Dead-letter entry not found")

    new_task_id = queue.replay(task_id)
    if new_task_id is None:  # replayed concurrently by someone else
        raise HTTPException(status_code=409, detail="Dead-letter entry was already replayed")

    job_id = entry["job_id"]
    with get_engine().begin() as conn:
        conn.execute(
            text(
                "UPDATE jobs SET status = 'queued', current_stage = 'queued', error_message = NULL, "
                "completed_at = NULL WHERE id = :id"
            ),
            {"id": job_id},
        )
        conn.execute(
            text("UPDATE batches SET status = 'queued', completed_at = NULL WHERE id = :id"),
            {"id": job_id},
        )
    _job_progress[job_id] = 0.0
    _job_eta.pop(job_id, None)
    return {"message": "Replay queued", "job_id": job_id, "task_id": new_task_id}
