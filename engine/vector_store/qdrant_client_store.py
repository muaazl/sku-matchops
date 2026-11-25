"""Singleton Qdrant client lifecycle, retry/fallback policy, and collection introspection."""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from collections.abc import Callable
from concurrent import futures
from typing import Any

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

from engine.core import config

logger = logging.getLogger("matchops.vector_store.client")

TRANSIENT_NETWORK_ERRORS = (
    ResponseHandlingException,
    httpx.ConnectError,
    httpx.NetworkError,
    httpx.TimeoutException,
    socket.gaierror,
    ConnectionError,
    TimeoutError,
    OSError,
)

_NON_RETRYABLE_STATUS_CODES = frozenset({400, 401, 403, 404, 409, 422})

_NON_RETRYABLE_GRPC_CODE_NAMES = frozenset(
    {
        "NOT_FOUND",
        "INVALID_ARGUMENT",
        "ALREADY_EXISTS",
        "PERMISSION_DENIED",
        "UNAUTHENTICATED",
        "UNIMPLEMENTED",
        "FAILED_PRECONDITION",
    }
)

_REBINDABLE_CLIENT_METHODS = (
    "get_collections",
    "get_collection",
    "get_aliases",
    "update_collection_aliases",
    "collection_exists",
    "create_collection",
    "update_collection",
    "delete_collection",
    "create_payload_index",
    "upsert",
    "query_points",
    "query_batch_points",
    "scroll",
    "count",
)


def extract_error_code_name(exc: Exception) -> str | None:
    """Extracts an RPC-style status code name without importing transport modules."""
    code_attr = getattr(exc, "code", None)
    raw_code = code_attr() if callable(code_attr) else code_attr
    if raw_code is None:
        return None
    name = getattr(raw_code, "name", None)
    if isinstance(name, str):
        return name
    if isinstance(raw_code, str):
        return raw_code
    return None


def serialize_qdrant_model(model: Any) -> dict[str, Any] | None:
    """Serializes a Qdrant Pydantic model response into a plain dictionary."""
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    if hasattr(model, "dict"):
        return model.dict()
    return model if isinstance(model, dict) else None


class QdrantClientManager:
    """Manages the singleton QdrantClient connection and transient-retry policy."""

    _client: QdrantClient | None = None
    _prefer_grpc_override: bool | None = None
    _checked_collections: set[str] = set()
    _lock = threading.RLock()

    @classmethod
    def ensure_local_storage_dir(cls) -> None:
        """Creates local Qdrant data directory only when operating without a remote URL."""
        qdrant_url = str(getattr(config, "QDRANT_URL", "") or "").strip()
        qdrant_dir = str(getattr(config, "QDRANT_DATA_DIR", "") or "").strip()
        if not qdrant_url and qdrant_dir:
            os.makedirs(qdrant_dir, exist_ok=True)

    @classmethod
    def get_qdrant_client(
        cls, force_reconnect: bool = False, prefer_grpc: bool | None = None
    ) -> QdrantClient:
        """Returns or reinitializes the singleton QdrantClient instance."""
        with cls._lock:
            if prefer_grpc is not None:
                cls._prefer_grpc_override = prefer_grpc
            if cls._client is None or force_reconnect:
                old_client = cls._client
                use_grpc = (
                    cls._prefer_grpc_override
                    if cls._prefer_grpc_override is not None
                    else getattr(config, "QDRANT_PREFER_GRPC", True)
                )
                cls.ensure_local_storage_dir()
                cls._client = QdrantClient(
                    url=config.QDRANT_URL,
                    grpc_port=getattr(config, "QDRANT_GRPC_PORT", 6334),
                    prefer_grpc=use_grpc,
                    timeout=config.QDRANT_TIMEOUT,
                    check_compatibility=False,
                )
                if old_client is not None:
                    try:
                        old_client.close()
                    except Exception:
                        pass
            return cls._client

    @classmethod
    def close_qdrant_client(cls) -> None:
        """Closes and resets the cached singleton client and schema-check cache."""
        with cls._lock:
            if cls._client is not None:
                try:
                    cls._client.close()
                except Exception:
                    pass
                cls._client = None
            cls._checked_collections.clear()

    @classmethod
    def resolve_bound_method_name(cls, client: Any, func: Callable[..., Any]) -> str | None:
        """Identifies which QdrantClient method `func` is bound to for transparent rebinding."""
        if getattr(func, "__self__", None) is client:
            return getattr(func, "__name__", None)
        return next(
            (attr for attr in _REBINDABLE_CLIENT_METHODS if getattr(client, attr, None) is func),
            None,
        )

    @classmethod
    def execute_with_retry(
        cls,
        client: Any,
        operation_name: str,
        func: Callable[..., Any],
        *args: Any,
        max_retries: int = 5,
        initial_delay: float = 0.5,
        backoff_factor: float = 2.0,
        on_reconnect: Callable[[Any], None] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Executes a Qdrant call with retries, client rebinding, and gRPC-to-REST fallback."""
        delay = initial_delay
        last_error: Exception | None = None
        active_client = client
        bound_method_name = cls.resolve_bound_method_name(active_client, func)

        for attempt in range(1, max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last_error = exc
                if getattr(exc, "status_code", None) in _NON_RETRYABLE_STATUS_CODES:
                    raise

                code_name = extract_error_code_name(exc)
                if code_name in _NON_RETRYABLE_GRPC_CODE_NAMES:
                    raise

                err_msg = str(exc).lower()
                is_grpc_unavailable = code_name == "UNAVAILABLE"
                is_transient = (
                    is_grpc_unavailable
                    or code_name == "DEADLINE_EXCEEDED"
                    or isinstance(exc, TRANSIENT_NETWORK_ERRORS)
                    or "name resolution" in err_msg
                    or "temporary failure" in err_msg
                    or "connection" in err_msg
                    or "timeout" in err_msg
                    or "errno -3" in err_msg
                    or "eai_again" in err_msg
                )

                if not is_transient or attempt == max_retries:
                    logger.error(
                        "[QDRANT] Operation '%s' failed permanently on attempt %d/%d: %s",
                        operation_name,
                        attempt,
                        max_retries,
                        exc,
                    )
                    raise

                # Fall back seamlessly to HTTP REST if gRPC port (6334) is unreachable.
                if (
                    is_grpc_unavailable
                    and active_client is cls._client
                    and cls._prefer_grpc_override is not False
                    and getattr(config, "QDRANT_PREFER_GRPC", True)
                ):
                    logger.warning(
                        "[QDRANT] gRPC port unavailable during '%s'; falling back to HTTP REST client.",
                        operation_name,
                    )
                    active_client = cls.get_qdrant_client(force_reconnect=True, prefer_grpc=False)
                    if on_reconnect is not None:
                        on_reconnect(active_client)
                    if bound_method_name:
                        func = getattr(active_client, bound_method_name)
                    continue

                logger.warning(
                    "[QDRANT] Transient error during '%s' (attempt %d/%d): %s. Retrying in %.2fs...",
                    operation_name,
                    attempt,
                    max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
                delay *= backoff_factor

                # Refresh singleton client instance on repeated connection glitches.
                if attempt >= 2 and active_client is cls._client:
                    active_client = cls.get_qdrant_client(force_reconnect=True)
                    if on_reconnect is not None:
                        on_reconnect(active_client)
                    if bound_method_name:
                        func = getattr(active_client, bound_method_name)

        assert last_error is not None
        raise last_error

    @classmethod
    def list_collection_details(
        cls,
        client_getter: Callable[[], QdrantClient],
        retry_caller: Callable[..., Any],
    ) -> dict[str, Any]:
        """Lists all Qdrant collections and fetches serialized details in parallel."""
        client = client_getter()
        collections_resp = retry_caller("get_collections", client.get_collections)
        col_names = [c.name for c in collections_resp.collections]
        details: dict[str, Any] = {}

        if col_names and hasattr(client, "get_collection"):

            def _fetch_single(col_name: str) -> tuple[str, dict[str, Any] | None]:
                try:
                    info = retry_caller(
                        "get_collection", client_getter().get_collection, collection_name=col_name
                    )
                    return col_name, serialize_qdrant_model(info)
                except Exception:
                    return col_name, None

            with futures.ThreadPoolExecutor(max_workers=min(len(col_names), 6)) as pool:
                for col_name, payload in pool.map(_fetch_single, col_names):
                    if isinstance(payload, dict):
                        details[col_name] = payload

        return {"collections": col_names, "details": details}

    @classmethod
    def fetch_collection_details(
        cls,
        client_getter: Callable[[], QdrantClient],
        retry_caller: Callable[..., Any],
        collection_name: str,
    ) -> dict[str, Any] | None:
        """Fetches serialized collection metadata for a single Qdrant collection."""
        info = retry_caller(
            "get_collection",
            client_getter().get_collection,
            collection_name=collection_name,
        )
        return serialize_qdrant_model(info)
