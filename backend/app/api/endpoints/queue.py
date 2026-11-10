from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import Connection, text

from backend.app.core.security import Principal, get_principal, require_admin
from backend.app.db.session import get_db_connection
from backend.app.grpc_server import clear_job_progress, publish_job_event, set_job_progress
from backend.app.services.rules_service import get_rule_version
from backend.app.services.task_queue import TaskQueue, TaskQueueProtocol, get_task_queue

router = APIRouter()


def get_queue() -> TaskQueueProtocol:
    queue = get_task_queue()
    if queue is None:
        raise HTTPException(status_code=503, detail="Task queue is disabled (TASK_QUEUE_ENABLED).")
    return queue


# Backwards compatibility alias
_queue = get_queue


def _tenant_items(queue: TaskQueueProtocol, principal: Principal, limit: int = 100) -> list[dict]:
    """DLQ entries of the caller's tenant."""
    return queue.dlq_items(limit=limit, tenant_id=principal.tenant_id)


@router.get("/queue/dlq")
def list_dead_letters(
    limit: int = Query(default=100, ge=1, le=1000),
    principal: Principal = Depends(require_admin),
    queue: TaskQueueProtocol = Depends(get_queue),
):
    """The tenant's dead-lettered tasks: payload plus job_id, batch_id, error, attempts, failed_at."""
    try:
        return _tenant_items(queue, principal, limit=limit)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Task queue unavailable: {e}")


@router.post("/queue/dlq/{task_id}/replay")
def replay_dead_letter(
    task_id: str,
    principal: Principal = Depends(require_admin),
    queue: TaskQueueProtocol = Depends(get_queue),
    db: Connection = Depends(get_db_connection),
):
    """Re-queues a dead-lettered task with a fresh retry budget and marks its job queued again."""
    try:
        entry = next(
            (e for e in _tenant_items(queue, principal, limit=10000) if e["task_id"] == task_id),
            None,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Task queue unavailable: {e}")

    if entry is None:
        raise HTTPException(status_code=404, detail="Dead-letter entry not found")

    job_id = entry["job_id"]
    payload = entry.get("payload") or {}
    batch_id = entry.get("batch_id") or payload.get("batch_id") or job_id

    # 1. Update SQL state with tenant-scoping first
    res = db.execute(
        text(
            "UPDATE jobs SET status = 'queued', current_stage = 'queued', error_message = NULL, "
            "completed_at = NULL WHERE id = :id AND tenant_id = :tenant_id"
        ),
        {"id": job_id, "tenant_id": principal.tenant_id},
    )
    if res.rowcount == 0:
        raise HTTPException(status_code=404, detail="Job record not found for tenant")

    db.execute(
        text(
            "UPDATE batches SET status = 'queued', completed_at = NULL WHERE id = :id AND tenant_id = :tenant_id"
        ),
        {"id": batch_id, "tenant_id": principal.tenant_id},
    )
    db.commit()

    # 2. Replay in Redis with error compensation
    try:
        new_task_id = queue.replay(task_id, tenant_id=principal.tenant_id)
    except Exception as e:
        try:
            db.execute(
                text(
                    "UPDATE jobs SET status = 'failed', current_stage = 'failed', error_message = 'Replay queue failed' "
                    "WHERE id = :id AND tenant_id = :tenant_id"
                ),
                {"id": job_id, "tenant_id": principal.tenant_id},
            )
            db.execute(
                text(
                    "UPDATE batches SET status = 'failed' WHERE id = :id AND tenant_id = :tenant_id"
                ),
                {"id": batch_id, "tenant_id": principal.tenant_id},
            )
            db.commit()
        except Exception:
            pass
        raise HTTPException(status_code=503, detail=f"Task queue replay failed: {e}")

    if new_task_id is None:  # replayed concurrently by someone else
        try:
            db.execute(
                text(
                    "UPDATE jobs SET status = 'failed', current_stage = 'failed', error_message = 'Concurrent replay detected' "
                    "WHERE id = :id AND tenant_id = :tenant_id"
                ),
                {"id": job_id, "tenant_id": principal.tenant_id},
            )
            db.commit()
        except Exception:
            pass
        raise HTTPException(status_code=409, detail="Dead-letter entry was already replayed")

    clear_job_progress(job_id)
    set_job_progress(job_id, pct=0.0, stage="queued", tenant_id=principal.tenant_id)
    publish_job_event(
        principal.tenant_id,
        {"id": job_id, "status": "queued", "current_stage": "queued", "progress_pct": 0.0},
    )
    return {"message": "Replay queued", "job_id": job_id, "task_id": new_task_id}


@router.get("/system/status")
def system_status(
    principal: Principal = Depends(get_principal),
    db: Connection = Depends(get_db_connection),
):
    """Operational counters for the portal health strip, scoped to the caller's tenant.
    Queue depths span every tenant, so only platform admins receive them."""
    processed = db.execute(
        text("SELECT COUNT(*) FROM processed_skus WHERE tenant_id = :t"),
        {"t": principal.tenant_id},
    ).scalar()

    queue_depth, dlq_size, queue_online = None, None, False
    try:
        queue = get_task_queue()
        if queue is not None:
            dlq_size = queue.dlq_count(principal.tenant_id)
            if principal.is_platform_admin:
                queue_depth = queue.stats()
            queue_online = True
    except Exception:
        pass  # Redis down: the strip shows the queue as offline

    return {
        "tenant_id": principal.tenant_id,
        "processed_skus": processed,
        "rule_version": get_rule_version(principal.tenant_id),
        "queue_online": queue_online,
        "queue_depth": queue_depth,
        "dlq_size": dlq_size,
    }
