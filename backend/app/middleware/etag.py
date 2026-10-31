import hashlib

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Responses at or above this size skip ETag computation entirely, to avoid
# buffering large bodies (catalog search pages, CSV/XLSX exports, log dumps)
# into memory just to MD5-hash them.
ETAG_MAX_BODY_SIZE = 256 * 1024  # 256 KB


class ETagMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # We only apply ETags to GET requests
        if request.method != "GET":
            return await call_next(request)

        response = await call_next(request)

        # Only ETag successful 200 OK responses
        if response.status_code != 200:
            return response

        # Skip large or unknown-size responses (e.g. a future streaming
        # response with no content-length) rather than buffering them blindly.
        content_length = response.headers.get("content-length")
        try:
            known_size = int(content_length) if content_length is not None else None
        except ValueError:
            known_size = None
        if known_size is None or known_size > ETAG_MAX_BODY_SIZE:
            return response

        # Capture response body
        body_bytes = b""
        if hasattr(response, "body_iterator"):
            response_body = [section async for section in response.body_iterator]

            async def async_iter():
                for chunk in response_body:
                    yield chunk

            response.body_iterator = async_iter()
            body_bytes = b"".join(response_body)
        elif hasattr(response, "body"):
            body_bytes = response.body

        # Calculate MD5 hash for ETag
        etag = f'W/"{hashlib.md5(body_bytes).hexdigest()}"'

        # Check If-None-Match header
        if_none_match = request.headers.get("if-none-match")
        if if_none_match and if_none_match == etag:
            # Return 304 Not Modified and preserve headers (including CORS)
            headers = dict(response.headers)
            headers["etag"] = etag
            headers.pop("content-length", None)
            headers.pop("content-type", None)
            return Response(status_code=304, headers=headers)

        # Set ETag header on original response
        response.headers["etag"] = etag
        return response
