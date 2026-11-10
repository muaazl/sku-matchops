"""Tests for Remediation Step 7:
- OBS-01: Prometheus HTTP metrics recorded on requests
- OBS-02: Structured log formats and context propagation with tenant_id
- Deep readiness probe (/ready) checking DB, Redis, and Engine
- Distributed Redis rate limiter with graceful fail-open
"""

import json
import logging
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException, Response
from starlette.requests import Request

import backend.app.core.observability as b_obs
import engine.core.observability as e_obs
from backend.app.api.routes import ready as ready_endpoint
from backend.app.core.observability import (
    HTTP_REQUESTS_TOTAL,
    JsonFormatter,
    TraceIdFilter,
    TraceIdMiddleware,
    trace_context,
)
from backend.app.core.rate_limit import RateLimiter
from backend.app.core.security import Principal


def test_trace_context_and_tenant_id_propagation():
    """Verify trace_context sets and restores both trace_id and tenant_id."""
    with trace_context("trace-abc-123", tenant_id="tenant-xyz-999"):
        assert b_obs.get_trace_id() == "trace-abc-123"
        assert b_obs.get_tenant_id() == "tenant-xyz-999"

    assert b_obs.get_trace_id() == "-"
    assert b_obs.get_tenant_id() == "-"


def test_log_filter_and_json_formatter_include_tenant():
    """Verify TraceIdFilter and JsonFormatter attach trace_id and tenant_id."""
    logger = logging.getLogger("test_logger")
    record = logger.makeRecord(
        name="test_logger",
        level=logging.INFO,
        fn="test_file.py",
        lno=42,
        msg="Test audit log message",
        args=(),
        exc_info=None,
    )

    with trace_context("trace-fixed-1", tenant_id="tenant-acme"):
        f = TraceIdFilter()
        f.filter(record)
        assert record.trace_id == "trace-fixed-1"
        assert record.tenant_id == "tenant-acme"

        formatter = JsonFormatter()
        formatted_json = formatter.format(record)
        entry = json.loads(formatted_json)
        assert entry["trace_id"] == "trace-fixed-1"
        assert entry["tenant_id"] == "tenant-acme"
        assert entry["message"] == "Test audit log message"


def test_engine_observability_tenant_context():
    """Verify engine.core.observability sets and formats tenant_id."""
    with e_obs.trace_context("eng-trace-1", tenant_id="eng-tenant-2"):
        assert e_obs.TRACE_ID.get() == "eng-trace-1"
        assert e_obs.TENANT_ID.get() == "eng-tenant-2"

        logger = logging.getLogger("engine_test")
        record = logger.makeRecord(
            name="engine_test",
            level=logging.INFO,
            fn="engine.py",
            lno=10,
            msg="Engine log",
            args=(),
            exc_info=None,
        )
        filt = e_obs.TraceIdFilter()
        filt.filter(record)
        assert record.trace_id == "eng-trace-1"
        assert record.tenant_id == "eng-tenant-2"


def test_prometheus_http_metrics_recorded():
    """Verify TraceIdMiddleware records HTTP_REQUESTS_TOTAL and latency."""
    # Synchronously test dispatch inner helper or test using mock Request & call_next

    async def _test():
        middleware = TraceIdMiddleware(app=MagicMock())
        req = MagicMock(spec=Request)
        req.headers = {"x-request-id": "req-999", "x-tenant-id": "tenant-corp"}
        req.url.path = "/api/v1/jobs/12345/cancel"
        req.method = "POST"

        resp = Response(content="ok", status_code=200)

        async def mock_call_next(r):
            # Verify context vars inside request pipeline
            assert b_obs.get_trace_id() == "req-999"
            assert b_obs.get_tenant_id() == "tenant-corp"
            return resp

        before_count = HTTP_REQUESTS_TOTAL.labels(
            method="POST", path="/api/v1/jobs/{id}/cancel", status="200"
        )._value.get()

        out_resp = await middleware.dispatch(req, mock_call_next)
        assert out_resp.headers["X-Request-ID"] == "req-999"

        after_count = HTTP_REQUESTS_TOTAL.labels(
            method="POST", path="/api/v1/jobs/{id}/cancel", status="200"
        )._value.get()
        assert after_count == before_count + 1

    # Run the coroutine with a clean mock or asyncio runner
    coro = _test()
    try:
        coro.send(None)
    except StopIteration:
        pass


def test_readiness_probe_success():
    """Verify /ready returns 200 when DB, Redis, and Engine are all healthy."""
    response = Response()
    with (
        patch("backend.app.api.routes.get_engine") as mock_engine,
        patch("redis.Redis.from_url") as mock_redis_cls,
        patch("backend.app.api.routes.get_engine_health") as mock_health,
    ):
        # DB ok
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn

        # Redis ok
        mock_r = MagicMock()
        mock_r.ping.return_value = True
        mock_redis_cls.return_value = mock_r

        # Engine ok
        mock_health.return_value = {"all_models_ready": True, "loaded_domains": ["market"]}

        result = ready_endpoint(response)
        assert result["status"] == "ready"
        assert result["db"] is True
        assert result["redis"] is True
        assert response.status_code == 200


def test_readiness_probe_failures():
    """Verify /ready returns 503 if any core dependency fails."""
    response = Response()

    # Case 1: Redis down
    with (
        patch("backend.app.api.routes.get_engine") as mock_engine,
        patch("redis.Redis.from_url") as mock_redis_cls,
        patch("backend.app.api.routes.get_engine_health") as mock_health,
    ):
        mock_engine.return_value.connect.return_value.__enter__.return_value = MagicMock()
        mock_r = MagicMock()
        mock_r.ping.side_effect = ConnectionError("Redis unreachable")
        mock_redis_cls.return_value = mock_r
        mock_health.return_value = {"all_models_ready": True}

        result = ready_endpoint(response)
        assert result["status"] == "not_ready"
        assert result["redis"] is False
        assert response.status_code == 503

    # Case 2: DB down
    response2 = Response()
    with (
        patch("backend.app.api.routes.get_engine") as mock_engine,
        patch("redis.Redis.from_url") as mock_redis_cls,
        patch("backend.app.api.routes.get_engine_health") as mock_health,
    ):
        mock_engine.return_value.connect.side_effect = Exception("DB Connection timeout")
        mock_r = MagicMock()
        mock_r.ping.return_value = True
        mock_redis_cls.return_value = mock_r
        mock_health.return_value = {"all_models_ready": True}

        result = ready_endpoint(response2)
        assert result["status"] == "not_ready"
        assert result["db"] is False
        assert response2.status_code == 503


def test_rate_limiter_allows_and_blocks():
    """Verify RateLimiter allows up to max_requests and raises HTTP 429 when exceeded."""
    limiter = RateLimiter(max_requests=5, window_seconds=60)

    req = MagicMock(spec=Request)
    req.url.path = "/match"
    principal = Principal(
        username="alice", role="reviewer", tenant_id="tenant-abc", home_tenant="tenant-abc"
    )

    mock_redis = MagicMock()
    pipe = MagicMock()
    mock_redis.pipeline.return_value = pipe

    with patch("backend.app.core.rate_limit._get_rate_limit_redis", return_value=mock_redis):
        # 1. Under limit (count = 3)
        pipe.execute.return_value = [3, True]
        limiter(req, principal)  # Should not raise

        # 2. Over limit (count = 6 > 5)
        pipe.execute.return_value = [6, True]
        with pytest.raises(HTTPException) as exc_info:
            limiter(req, principal)
        assert exc_info.value.status_code == 429
        assert "Rate limit exceeded" in exc_info.value.detail
        assert "Retry-After" in exc_info.value.headers


def test_rate_limiter_fails_open_on_redis_error():
    """Verify RateLimiter fails open gracefully if Redis raises an error."""
    limiter = RateLimiter(max_requests=5, window_seconds=60)
    req = MagicMock(spec=Request)
    req.url.path = "/match"
    principal = Principal(
        username="alice", role="reviewer", tenant_id="tenant-abc", home_tenant="tenant-abc"
    )

    # Redis client is None (disconnected)
    with patch("backend.app.core.rate_limit._get_rate_limit_redis", return_value=None):
        limiter(req, principal)  # Should fail open without exception

    # Redis client raises on execute
    mock_redis = MagicMock()
    mock_redis.pipeline.side_effect = RuntimeError("Redis timeout")
    with patch("backend.app.core.rate_limit._get_rate_limit_redis", return_value=mock_redis):
        limiter(req, principal)  # Should fail open without exception
