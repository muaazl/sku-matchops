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
                    f"WARNING: Timeout waiting for service {host}:{port}. Attempting to proceed anyway...",
                    flush=True,
                )
                return False
            time.sleep(1.0)


def main():
    print("=" * 60, flush=True)
    print("SKU MatchOps Backend API Gateway starting...", flush=True)
    print("=" * 60, flush=True)

    # Launch uvicorn for the Backend API Gateway
    port = os.getenv("PORT", "8000")
    print(f"Starting SKU MatchOps Backend API Gateway on 0.0.0.0:{port}...", flush=True)
    print("=" * 60, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()

    os.execvp("uvicorn", ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", port])


if __name__ == "__main__":
    main()
