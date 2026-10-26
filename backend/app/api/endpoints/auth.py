from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from sqlalchemy import Connection, text
from sqlalchemy.exc import IntegrityError

from backend.app.core.db import get_db_connection
from backend.app.core.security import (
    Principal,
    create_access_token,
    get_principal,
    hash_password,
    require_admin,
    verify_password,
)

router = APIRouter()  # public
users_router = APIRouter()  # authenticated


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    tenant_id: str


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_.@-]+$")
    password: str = Field(..., min_length=8, max_length=256)
    role: Literal["admin", "rule_engineer", "reviewer"]


def create_user(conn: Connection, username: str, password: str, role: str, tenant_id: str) -> int:
    """Inserts a user; raises IntegrityError if the username is taken."""
    return conn.execute(
        text(
            "INSERT INTO users (username, password_hash, role, tenant_id) "
            "VALUES (:u, :p, :r, :t) RETURNING id"
        ),
        {"u": username, "p": hash_password(password), "r": role, "t": tenant_id},
    ).scalar()


@router.post("/auth/token", response_model=TokenResponse)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Connection = Depends(get_db_connection)):
    """OAuth2 password flow: exchange username/password for a bearer JWT."""
    user = (
        db.execute(
            text(
                "SELECT username, password_hash, role, tenant_id FROM users "
                "WHERE username = :u AND is_active = 1"
            ),
            {"u": form.username},
        )
        .mappings()
        .fetchone()
    )
    if not user or not verify_password(form.password, user["password_hash"]):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(user["username"], user["role"], user["tenant_id"])
    return TokenResponse(access_token=token, role=user["role"], tenant_id=user["tenant_id"])


@users_router.get("/auth/me")
def me(principal: Principal = Depends(get_principal)):
    return {
        "username": principal.username,
        "role": principal.role,
        "tenant_id": principal.tenant_id,
        "platform_admin": principal.is_platform_admin,
    }


@users_router.get("/auth/users")
def list_users(
    db: Connection = Depends(get_db_connection), principal: Principal = Depends(require_admin)
):
    rows = db.execute(
        text(
            "SELECT id, username, role, tenant_id, is_active, created_at FROM users "
            "WHERE tenant_id = :t ORDER BY id"
        ),
        {"t": principal.tenant_id},
    ).mappings()
    return [dict(r) for r in rows]


@users_router.post("/auth/users", status_code=201)
def add_user(
    user: UserCreate,
    db: Connection = Depends(get_db_connection),
    principal: Principal = Depends(require_admin),
):
    """Creates a user in the caller's tenant (platform admins can target one via X-Tenant-ID)."""
    try:
        user_id = create_user(db, user.username, user.password, user.role, principal.tenant_id)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Username already exists.")
    return {
        "id": user_id,
        "username": user.username,
        "role": user.role,
        "tenant_id": principal.tenant_id,
    }
