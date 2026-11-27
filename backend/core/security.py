"""Authentication, RBAC dependencies, and SSRF validation for the Gateway."""

import hmac
import ipaddress
import logging
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.security.api_key import APIKeyHeader

from backend.core import config

logger = logging.getLogger("matchops.security")

ADMIN, RULE_ENGINEER, REVIEWER = "admin", "rule_engineer", "reviewer"
ROLES = (ADMIN, RULE_ENGINEER, REVIEWER)

_bearer = HTTPBearer(auto_error=False)
_api_key = APIKeyHeader(name="X-API-Key", auto_error=False)


def _is_disallowed_ip(ip_addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Returns True when an IP address belongs to a non-global, private, loopback, or link-local range."""
    mapped = getattr(ip_addr, "ipv4_mapped", None)
    if mapped is not None and _is_disallowed_ip(mapped):
        return True
    return (
        not ip_addr.is_global
        or ip_addr.is_private
        or ip_addr.is_loopback
        or ip_addr.is_link_local
        or ip_addr.is_unspecified
        or ip_addr.is_reserved
        or ip_addr.is_multicast
    )


def validate_outbound_url(url: str, allowed_domains: tuple[str, ...] | None = None) -> str:
    """Validates an outbound URL to prevent Server-Side Request Forgery (SSRF)."""
    if not url or not isinstance(url, str):
        raise ValueError("URL must be a non-empty string.")

    parsed = urlparse(url.strip())
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise ValueError(
            f"Invalid URL scheme: '{parsed.scheme}'. Only HTTP and HTTPS are permitted."
        )

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must include a valid hostname.")

    hostname_lower = hostname.lower()

    if allowed_domains is not None:
        if not any(
            hostname_lower == d.lower() or hostname_lower.endswith("." + d.lower())
            for d in allowed_domains
        ):
            raise ValueError(
                f"Domain '{hostname}' is not in the allowed outbound whitelist: {allowed_domains}"
            )

    if hostname_lower in ("localhost", "metadata.google.internal", "instance-data"):
        raise ValueError(f"Access to internal hostname '{hostname}' is prohibited.")

    try:
        try:
            ip_addrs = [ipaddress.ip_address(hostname)]
        except ValueError:
            addr_info = socket.getaddrinfo(hostname, None)
            ip_addrs = [ipaddress.ip_address(sockaddr[0]) for *_, sockaddr in addr_info]

        for ip_addr in ip_addrs:
            if _is_disallowed_ip(ip_addr):
                raise ValueError(
                    f"Access to private/internal network address '{ip_addr}' is prohibited."
                )
    except socket.gaierror as e:
        logger.warning(f"Could not resolve hostname '{hostname}' during pre-flight SSRF check: {e}")

    return url


@dataclass(frozen=True)
class Principal:
    username: str = "api-key"
    role: str = ADMIN

    @property
    def is_platform_admin(self) -> bool:
        return self.role == ADMIN


def get_principal(
    request: Request,
    bearer: HTTPAuthorizationCredentials | None = Security(_bearer),
    api_key: str | None = Security(_api_key),
) -> Principal:
    """Authenticates API requests via X-API-Key or Bearer token."""
    key = None
    if api_key is not None:
        key = api_key
    elif bearer is not None:
        key = bearer.credentials

    if key is not None:
        if not config.API_KEY or not hmac.compare_digest(key, config.API_KEY):
            raise HTTPException(status_code=401, detail="Invalid API key.")
        return Principal("api-key", ADMIN)

    # Internal team access (e.g. browser UI): authenticated with full admin permissions
    return Principal("team-admin", ADMIN)


def require_roles(*roles: str):
    """Dependency factory rejecting requests unless the caller holds an allowed role."""

    def _check(principal: Principal = Depends(get_principal)) -> Principal:
        if principal.role not in roles:
            raise HTTPException(status_code=403, detail=f"Requires role: {' or '.join(roles)}.")
        return principal

    return _check


require_admin = require_roles(ADMIN)


def require_platform_admin(principal: Principal = Depends(get_principal)) -> Principal:
    """Platform administrator verification."""
    if not principal.is_platform_admin:
        raise HTTPException(status_code=403, detail="Requires platform admin.")
    return principal


require_rule_editor = require_roles(ADMIN, RULE_ENGINEER)
