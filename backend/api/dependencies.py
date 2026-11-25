"""Shared FastAPI request dependencies."""

from fastapi import Depends, HTTPException, Request

from backend.core.security import Principal, get_principal
from backend.services import rate_limit_service


class RateLimiter:
    """FastAPI dependency enforcing Redis-backed fixed-window request quotas."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    def __call__(
        self,
        request: Request,
        principal: Principal = Depends(get_principal),
    ) -> None:
        subject = (
            principal.username
            if principal
            else (request.client.host if request.client else "unknown")
        )
        allowed, retry_after = rate_limit_service.check_rate_limit(
            subject=subject,
            endpoint=request.url.path,
            max_requests=self.max_requests,
            window_seconds=self.window_seconds,
        )
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Try again later.",
                headers={"Retry-After": str(retry_after)},
            )
