import json
import logging
import sqlite3
import time
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from engine.rules_engine.db.seed_rules import DB_PATH

logger = logging.getLogger("matchops.middleware")

SENSITIVE_SUBSTRINGS = ('token', 'auth', 'bearer', 'password', 'secret', 'key', 'x-api-key')

def _redact_sensitive_data(obj):
    """Recursively redacts sensitive keys in dictionaries and lists."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                _redact_sensitive_data(v)
            else:
                key_lower = k.lower()
                if any(sub in key_lower for sub in SENSITIVE_SUBSTRINGS):
                    obj[k] = '***REDACTED***'
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                _redact_sensitive_data(item)


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Determine if we should log this request
        path = request.url.path
        method = request.method
        
        # Only log matcher, classify, and pipeline calls, exclude GETs (polling)
        log_paths = ('/match', '/classify', '/pipeline')
        should_log = path.startswith(log_paths) and method != "GET"
        
        if not should_log:
            return await call_next(request)

        start_time = time.time()
        req_id = str(uuid.uuid4())
        
        # Read body only for logged requests.
        body = await request.body()
        
        # Restore body for the next handlers
        async def receive():
            return {"type": "http.request", "body": body}
        request._receive = receive
        
        payload_redacted = ""
        ip_address = ""
        headers_json = ""
        query_params_json = ""
        
        # 1. Get client IP (support proxies/cloudflare tunnel)
        ip_address = request.headers.get("x-forwarded-for")
        if not ip_address:
            ip_address = request.headers.get("x-real-ip")
        if not ip_address:
            ip_address = request.client.host if request.client else "127.0.0.1"
        if ip_address and "," in ip_address:
            ip_address = ip_address.split(",")[0].strip()

        # 2. Capture and redact request headers
        try:
            headers_dict = dict(request.headers)
            for key in list(headers_dict.keys()):
                key_lower = key.lower()
                if any(sub in key_lower for sub in SENSITIVE_SUBSTRINGS):
                    headers_dict[key] = "***REDACTED***"
            headers_json = json.dumps(headers_dict)
        except Exception:
            headers_json = "{}"

        # 3. Capture query parameters
        try:
            query_params_dict = dict(request.query_params)
            query_params_json = json.dumps(query_params_dict)
        except Exception:
            query_params_json = "{}"

        # 4. Redact payload/body
        try:
            if body:
                payload_json = json.loads(body.decode('utf-8'))
                _redact_sensitive_data(payload_json)
                payload_redacted = json.dumps(payload_json)
        except Exception:
            payload_redacted = "<non-json body or decode error>"
            
        # Process the request
        response = await call_next(request)
            
        duration_ms = int((time.time() - start_time) * 1000)
        status_code = response.status_code
        
        # 5. Capture response body safely
        response_json = ""
        try:
            if hasattr(response, "body_iterator"):
                response_body = [section async for section in response.body_iterator]
                async def async_iter():
                    for chunk in response_body:
                        yield chunk
                response.body_iterator = async_iter()
                body_bytes = b"".join(response_body)
                response_json = body_bytes.decode('utf-8', errors='replace')
            elif hasattr(response, "body"):
                response_json = response.body.decode('utf-8', errors='replace')
                
            # Safely redact response JSON
            if response_json:
                try:
                    resp_data = json.loads(response_json)
                    _redact_sensitive_data(resp_data)
                    response_json = json.dumps(resp_data)
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Failed to capture response body: {e}")
        
        # Write to DB
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                INSERT INTO api_requests (id, method, path, payload_json_redacted, response_json, status_code, duration_ms, ip_address, headers_json, query_params_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (req_id, method, path, payload_redacted, response_json, status_code, duration_ms, ip_address, headers_json, query_params_json)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to log API request: {e}")
            
        return response
