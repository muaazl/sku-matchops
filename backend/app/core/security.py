"""
Authentication, RBAC and tenant resolution for the Gateway.

A request is resolved to a Principal(username, role, tenant_id, home_tenant) from, in order:
  1. `Authorization: Bearer <JWT>`  - role and home tenant come from the signed claims
  2. `X-API-Key`                     - trusted service caller: platform admin
  3. nothing, with AUTH_ENABLED off  - dev mode: platform admin
A platform admin is an admin whose home tenant is 'default' (the operator). Only platform admins
may act on another tenant via X-Tenant-ID or use platform-wide endpoints (shared catalog, models,
raw request log). X-Tenant-ID defaults to the caller's home tenant.
"""

import hashlib
import hmac
import logging
import re
import secrets
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

import jwt
from fastapi import Depends, Header, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.security.api_key import APIKeyHeader

from backend.app.core import config

logger = logging.getLogger("matchops.security")

ADMIN, RULE_ENGINEER, REVIEWER = "admin", "rule_engineer", "reviewer"
ROLES = (ADMIN, RULE_ENGINEER, REVIEWER)
_TENANT_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

_bearer = HTTPBearer(auto_error=False)
_api_key = APIKeyHeader(name="X-API-Key", auto_error=False)
_ephemeral_secret = secrets.token_urlsafe(48)


@dataclass(frozen=True)
class Principal:
    username: str
    role: str
    tenant_id: str  # tenant this request acts on
    home_tenant: str = config.DEFAULT_TENANT  # tenant the identity belongs to

    @property
    def is_platform_admin(self) -> bool:
        return self.role == ADMIN and self.home_tenant == config.DEFAULT_TENANT


# --- Passwords (PBKDF2-SHA256, stdlib) -------------------------------------------------------
_PBKDF2_ITERATIONS = 200_000


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), _PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iterations, salt, expected = stored.split("$")
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), int(iterations))
    except (ValueError, AttributeError):
        return False
    return hmac.compare_digest(digest.hex(), expected)


# --- JWT -----------------------------------------------------------------------------------
def _secret() -> str:
    if config.JWT_SECRET:
        return config.JWT_SECRET
    if config.AUTH_ENABLED:
        raise RuntimeError("JWT_SECRET must be set when AUTH_ENABLED=true")
    return _ephemeral_secret


def create_access_token(username: str, role: str, tenant_id: str) -> str:
    now = datetime.now(UTC)
    claims = {
        "sub": username,
        "role": role,
        "tenant_id": tenant_id,
        "iat": now,
        "exp": now + timedelta(minutes=config.JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(claims, _secret(), algorithm=config.JWT_ALGORITHM)


def _decode(token: str) -> Principal:
    try:
        claims = jwt.decode(
            token, _secret(), algorithms=[config.JWT_ALGORITHM], options={"require": ["exp", "sub"]}
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired.")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token.")
    if claims.get("role") not in ROLES or not claims.get("tenant_id"):
        raise HTTPException(status_code=401, detail="Invalid token claims.")
    return Principal(claims["sub"], claims["role"], claims["tenant_id"], claims["tenant_id"])


# --- Dependencies ----------------------------------------------------------------------------
def _tenant_header(x_tenant_id: str | None) -> str | None:
    if x_tenant_id is None:
        return None
    if not _TENANT_RE.match(x_tenant_id):
        raise HTTPException(status_code=400, detail="Invalid X-Tenant-ID.")
    return x_tenant_id


def get_principal(
    bearer: HTTPAuthorizationCredentials | None = Security(_bearer),
    api_key: str | None = Security(_api_key),
    x_tenant_id: str | None = Header(default=None),
) -> Principal:
    tenant_header = _tenant_header(x_tenant_id)

    if bearer is not None:
        principal = _decode(bearer.credentials)
    elif api_key is not None:
        if not hmac.compare_digest(api_key, config.API_KEY):
            raise HTTPException(status_code=401, detail="Invalid API key.")
        principal = Principal("api-key", ADMIN, config.DEFAULT_TENANT)
    elif not config.AUTH_ENABLED:
        principal = Principal("anonymous-dev", ADMIN, config.DEFAULT_TENANT)
    else:
        raise HTTPException(
            status_code=401, detail="Not authenticated.", headers={"WWW-Authenticate": "Bearer"}
        )

    if tenant_header and tenant_header != principal.home_tenant:
        if not principal.is_platform_admin:
            raise HTTPException(status_code=403, detail="Cannot act on another tenant.")
        return replace(principal, tenant_id=tenant_header)
    return principal


def require_roles(*roles: str):
    """Dependency factory: 403 unless the caller has one of `roles`."""

    def _check(principal: Principal = Depends(get_principal)) -> Principal:
        if principal.role not in roles:
            raise HTTPException(status_code=403, detail=f"Requires role: {' or '.join(roles)}.")
        return principal

    return _check


require_admin = require_roles(ADMIN)


def require_platform_admin(principal: Principal = Depends(get_principal)) -> Principal:
    """Admin of the operator ('default') tenant: needed for endpoints shared by all tenants."""
    if not principal.is_platform_admin:
        raise HTTPException(status_code=403, detail="Requires platform admin.")
    return principal


require_rule_editor = require_roles(ADMIN, RULE_ENGINEER)
