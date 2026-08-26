import os
import sys
import time
import socket

def wait_for_service(host: str, port: int, timeout: float = 60.0):
    """Waits for a service to become available on host:port."""
    start_time = time.time()
    print(f"Waiting for service {host}:{port}...", flush=True)
    while True:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                print(f"Service {host}:{port} is online!", flush=True)
                return True
        except (socket.timeout, ConnectionRefusedError):
            if time.time() - start_time > timeout:
                print(f"WARNING: Timeout waiting for service {host}:{port}. Attempting to proceed anyway...", flush=True)
                return False
            time.sleep(1.0)

def main():
    print("=" * 60, flush=True)
    print("SKU MatchOps Backend API Gateway starting...", flush=True)
    print("=" * 60, flush=True)
    
    meili_url = os.getenv("MEILI_URL", "http://meilisearch:7700")
    
    def parse_url(url):
        parts = url.replace("http://", "").replace("https://", "").split("/")
        host_port = parts[0].split(":")
        host = host_port[0]
        port = int(host_port[1]) if len(host_port) > 1 else 80
        return host, port

    try:
        m_host, m_port = parse_url(meili_url)
        wait_for_service(m_host, m_port, timeout=45.0)
    except Exception as e:
        print(f"Could not parse or connect to MEILI_URL={meili_url}: {e}", flush=True)

    # Launch uvicorn for the Backend API Gateway
    port = os.getenv("PORT", "8000")
    print("=" * 60, flush=True)
    print(f"Starting SKU MatchOps Backend API Gateway on 0.0.0.0:{port}...", flush=True)
    print("=" * 60, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    
    os.execvp("uvicorn", ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", port])

if __name__ == "__main__":
    main()
