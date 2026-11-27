"""HTTP ETag middleware computing weak MD5 ETags for cacheable GET responses."""

import hashlib
from collections.abc import AsyncIterator

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

ETAG_MAX_BODY_SIZE = 256 * 1024  # 256 KB


async def _chain_chunks(
    buffered: list[bytes], remaining: AsyncIterator[bytes]
) -> AsyncIterator[bytes]:
    for chunk in buffered:
        yield chunk
    async for chunk in remaining:
        yield chunk


def _etag_matches(header_value: str | None, etag: str) -> bool:
    if not header_value:
        return False
    stripped = header_value.strip()
    if stripped == "*":
        return True
    target_opaque = etag.removeprefix("W/").strip()
    for candidate in stripped.split(","):
        if candidate.strip().removeprefix("W/").strip() == target_opaque:
            return True
    return False


class ETagMiddleware(BaseHTTPMiddleware):
    """Adds weak MD5 ETags and returns 304 Not Modified when `If-None-Match` matches."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)

        if request.method != "GET" or response.status_code != 200:
            return response
        if request.url.path.endswith("/stream"):
            return response
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            return response

        content_length = response.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > ETAG_MAX_BODY_SIZE:
                    return response
            except ValueError:
                pass

        body_iterator = getattr(response, "body_iterator", None)
        if body_iterator is None:
            return response

        chunks: list[bytes] = []
        total_size = 0
        async for raw_chunk in body_iterator:
            chunk = raw_chunk if isinstance(raw_chunk, bytes) else raw_chunk.encode("utf-8")
            chunks.append(chunk)
            total_size += len(chunk)
            if total_size > ETAG_MAX_BODY_SIZE:
                headers = {
                    k: v for k, v in response.headers.items() if k.lower() != "content-length"
                }
                return StreamingResponse(
                    _chain_chunks(chunks, body_iterator),
                    status_code=response.status_code,
                    headers=headers,
                    media_type=response.media_type,
                )

        body = b"".join(chunks)
        digest = hashlib.md5(body, usedforsecurity=False).hexdigest()
        etag = f'W/"{digest}"'

        headers = {k: v for k, v in response.headers.items() if k.lower() != "content-length"}
        headers["ETag"] = etag

        if _etag_matches(request.headers.get("if-none-match"), etag):
            return Response(status_code=304, headers=headers)

        return Response(
            content=body,
            status_code=response.status_code,
            headers=headers,
            media_type=response.media_type,
        )
