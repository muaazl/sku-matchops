"""
SKU MatchOps Backend - Redis-Backed Distributed Rate Limiter
Prevents volumetric abuse and denial-of-service on computationally expensive endpoints
(/match, /classify, /pipeline, /rules/test-draft).
"""

import logging
import time

from fastapi import Depends, HTTPException, Request

from backend.app.core import config as backend_config
from backend.app.core.security import Principal, get_principal

logger = logging.getLogger("matchops.rate_limit")

_redis_client = None


def _get_rate_limit_redis():
    global _redis_client
    if _redis_client is None:
        try:
            import redis

            _redis_client = redis.Redis.from_url(
                backend_config.REDIS_URL,
                socket_connect_timeout=1.0,
                socket_timeout=1.0,
                decode_responses=True,
            )
        except Exception as e:
            logger.warning(f"Could not connect to Redis for rate limiting: {e}")
            return None
    return _redis_client


class RateLimiter:
    """
    Fixed-window counter rate limiter per caller (or IP if unauthenticated).
    Fails open gracefully if Redis is temporarily unavailable.
    """

    def __init__(self, max_requests: int = 120, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    def __call__(
        self,
        request: Request,
        principal: Principal = Depends(get_principal),
    ) -> None:
        client = _get_rate_limit_redis()
        if client is None:
            return  # Fail open if Redis unavailable

        # Identify subject: authenticated username/role, or client host
        subject = (
            principal.username
            if principal and principal.username
            else (request.client.host if request.client else "anonymous")
        )
        endpoint = request.url.path

        # Current time bucket
        current_bucket = int(time.time()) // self.window_seconds
        key = f"matchops:ratelimit:{subject}:{endpoint}:{current_bucket}"

        try:
            pipe = client.pipeline()
            pipe.incr(key)
            pipe.expire(key, self.window_seconds * 2)
            results = pipe.execute()
            count = results[0]

            if count > self.max_requests:
                retry_after = self.window_seconds - (int(time.time()) % self.window_seconds)
                logger.warning(
                    f"Rate limit exceeded for subject={subject} on {endpoint}: {count} > {self.max_requests}"
                )
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Maximum {self.max_requests} requests per {self.window_seconds}s.",
                    headers={"Retry-After": str(max(1, retry_after))},
                )
        except HTTPException:
            raise
        except Exception as err:
            logger.warning(f"Rate limiting check failed ({err}); failing open.")
