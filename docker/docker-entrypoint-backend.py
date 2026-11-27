import os
import socket
import sys
import time


def wait_for_service(host: str, port: int, timeout: float = 60.0):
    """Waits for a service to become available on host:port."""
    start_time = time.time()
    print(f"Waiting for service {host}:{port}...", flush=True)
    while True:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                print(f"Service {host}:{port} is online!", flush=True)
                return True
        except (TimeoutError, ConnectionRefusedError):
            if time.time() - start_time > timeout:
                print(
                    f"FATAL: Timeout waiting for service {host}:{port}.",
                    flush=True,
                )
                return False
            time.sleep(1.0)


def main():
    print("=" * 60, flush=True)
    print("SKU MatchOps Backend API Gateway starting...", flush=True)
    print("=" * 60, flush=True)

    # Wait for PostgreSQL
    pg_host = os.getenv("POSTGRES_HOST", "postgres")
    pg_port = int(os.getenv("POSTGRES_PORT", "5432"))
    if not wait_for_service(pg_host, pg_port, timeout=30.0):
        sys.exit(1)

    # Wait for Redis (required, no fallback)
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    parts = redis_url.replace("redis://", "").split("/")
    host_port = parts[0].split("@")[-1].split(":")
    r_host = host_port[0]
    r_port = int(host_port[1]) if len(host_port) > 1 else 6379
    if not wait_for_service(r_host, r_port, timeout=30.0):
        sys.exit(1)

    # Launch uvicorn for the Backend API Gateway (single process, no reload bloat)
    port = os.getenv("PORT", "8000")
    reload_env = os.getenv("UVICORN_RELOAD", "false").strip().lower()
    reload_enabled = reload_env in ("true", "1", "yes")

    print(f"Starting SKU MatchOps Backend API Gateway on 0.0.0.0:{port}...", flush=True)
    print("=" * 60, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()

    cmd = ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", port]
    if reload_enabled:
        cmd.extend(["--reload", "--reload-dir", "/app/backend"])

    os.execvp("uvicorn", cmd)


if __name__ == "__main__":
    main()
