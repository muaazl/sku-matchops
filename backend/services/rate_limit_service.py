"""Rate limiting policy service backed by Redis fixed-window counters."""

import logging

from backend.repositories import rate_limit_repository

logger = logging.getLogger("matchops.services.rate_limit")


def check_rate_limit(
    *,
    subject: str,
    endpoint: str,
    max_requests: int,
    window_seconds: int,
) -> tuple[bool, int]:
    """Evaluates whether `subject` is within the allowed request quota for `endpoint`."""
    counter = rate_limit_repository.increment_rate_limit_counter(
        subject=subject,
        endpoint=endpoint,
        window_seconds=window_seconds,
    )
    if counter is None:
        return True, 0
    count, retry_after = counter
    if count > max_requests:
        logger.warning(
            f"Rate limit exceeded for subject={subject} on {endpoint}: {count} > {max_requests}"
        )
        return False, retry_after
    return True, 0
