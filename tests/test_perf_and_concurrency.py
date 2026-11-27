"""Tests for Remediation Steps 5 & 6:
- PER-03: Pooled async Redis connections for SSE pubsub
- DIS-01: Distributed Redis lock for catalog cache builds
- PER-04: Memory protection against large response bodies in audit logging
- Schema: Uniqueness constraints and composite performance indexes
"""

import json
from unittest.mock import MagicMock, patch

import backend.repositories.catalog_lock_repository as lock_repo
import backend.repositories.job_progress_repository as jobs_mod
import backend.services.catalog_service as cat_svc
from backend.models.job_model import jobs as jobs_table
from backend.models.sku_model import processed_skus as processed_skus_table
from backend.repositories.job_progress_repository import get_async_redis_pool


def test_async_redis_pool_singleton():
    """Verify get_async_redis_pool creates and returns a singleton pool with max_connections=50."""
    with patch("redis.asyncio.ConnectionPool.from_url") as mock_from_url:
        mock_pool = MagicMock()
        mock_from_url.return_value = mock_pool

        jobs_mod._async_redis_pool = None
        try:
            pool1 = get_async_redis_pool()
            pool2 = get_async_redis_pool()

            assert pool1 is pool2
            mock_from_url.assert_called_once()
            _, kwargs = mock_from_url.call_args
            assert kwargs.get("max_connections") == 50
            assert kwargs.get("socket_connect_timeout") == 2.0
            assert kwargs.get("decode_responses") is True
        finally:
            jobs_mod._async_redis_pool = None


def test_catalog_build_distributed_redis_lock():
    """Verify trigger_build_cache uses cluster-wide distributed lock matchops:lock:catalog_build."""
    cat_svc._build_in_progress = False

    mock_redis = MagicMock()
    # Case 1: Successfully acquires lock
    mock_redis.set.return_value = True

    with patch.object(lock_repo, "get_catalog_lock_redis_client", return_value=mock_redis):
        started, msg = cat_svc.trigger_build_cache()
        assert started is True
        assert "initiated in background" in msg
        mock_redis.set.assert_called_once_with(lock_repo.BUILD_LOCK_KEY, "locked", nx=True, ex=1800)

        # Case 2: Concurrent call on same node while in progress
        started2, msg2 = cat_svc.trigger_build_cache()
        assert started2 is False
        assert "already in progress" in msg2

        # Reset in-progress flag
        cat_svc._build_in_progress = False

        # Case 3: Another node in the cluster holds the lock
        mock_redis.set.return_value = False
        started3, msg3 = cat_svc.trigger_build_cache()
        assert started3 is False
        assert "already in progress across cluster" in msg3


def test_catalog_build_lock_cleanup_on_completion():
    """Verify get_build_task_callable cleans up distributed lock on completion."""
    cat_svc._build_in_progress = True
    mock_redis = MagicMock()

    with (
        patch.object(lock_repo, "get_catalog_lock_redis_client", return_value=mock_redis),
        patch.object(cat_svc, "execute_background_cache_build") as mock_bg,
    ):
        task = cat_svc.get_build_task_callable()
        task()

        mock_bg.assert_called_once()
        assert cat_svc._build_in_progress is False
        mock_redis.delete.assert_called_once_with(lock_repo.BUILD_LOCK_KEY)


def test_audit_logging_omits_large_response_bodies():
    """Verify audit logging middleware truncates/omits response bodies exceeding 64KB."""
    from backend.api.middleware.audit_logging_middleware import format_audit_response

    # Case A: Content-Length header indicates > 64KB
    res_cl = format_audit_response(None, content_length=70000)
    assert res_cl == "<omitted: response body 70000 bytes > 64KB>"

    # Case B: In-memory/streamed body bytes exceed 64KB
    large_bytes = b"X" * 70000
    res_bytes = format_audit_response(large_bytes)
    assert res_bytes == "<omitted: response body 70000 bytes > 64KB>"

    # Case C: Response body is within limits and gets redacted
    small_json = json.dumps({"token": "secret123", "result": "success"}).encode("utf-8")
    res_small = format_audit_response(small_json)
    parsed = json.loads(res_small)
    assert parsed["token"] == "***REDACTED***"
    assert parsed["result"] == "success"


def test_schema_constraints_and_composite_indexes():
    """Verify schema defines uniqueness constraints and composite performance indexes."""
    # 1. processed_skus: unique constraint on (batch_id, sku_index, rule_version)
    sku_uq_names = [c.name for c in processed_skus_table.constraints if hasattr(c, "columns")]
    assert "uq_processed_skus_job_sku_rule" in sku_uq_names
    for c in processed_skus_table.constraints:
        if c.name == "uq_processed_skus_job_sku_rule":
            col_names = [col.name for col in c.columns]
            assert col_names == ["batch_id", "sku_index", "rule_version"]

    # 2. jobs: composite index on (status, started_at)
    job_idx_names = [idx.name for idx in jobs_table.indexes]
    assert "idx_jobs_status_started" in job_idx_names
    for idx in jobs_table.indexes:
        if idx.name == "idx_jobs_status_started":
            col_names = [col.name for col in idx.columns]
            assert col_names == ["status", "started_at"]

    # 3. processed_skus: composite index on (batch_id, created_at)
    sku_idx_names = [idx.name for idx in processed_skus_table.indexes]
    assert "idx_processed_skus_batch_created" in sku_idx_names
    for idx in processed_skus_table.indexes:
        if idx.name == "idx_processed_skus_batch_created":
            col_names = [col.name for col in idx.columns]
            assert col_names == ["batch_id", "created_at"]
