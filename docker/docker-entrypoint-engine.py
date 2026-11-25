import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path


def wait_for_service(host: str, port: int, timeout: float = 60.0):
    """Waits for a service to become available on host:port."""
    start_time = time.time()
    print(f"Waiting for database service {host}:{port}...", flush=True)
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


def sync_host_directory(host_dir: str, target_dir: str, label: str):
    """Syncs missing or updated files from host bind-mount into fast Docker volume."""
    host_p = Path(host_dir)
    target_p = Path(target_dir)
    if not host_p.exists():
        return
    synced = 0
    start = time.time()
    for src_file in host_p.rglob("*"):
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(host_p)
        dest_file = target_p / rel
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            s_stat = src_file.stat()
            if not dest_file.exists():
                shutil.copy2(src_file, dest_file)
                synced += 1
            else:
                d_stat = dest_file.stat()
                if s_stat.st_size != d_stat.st_size or s_stat.st_mtime > d_stat.st_mtime + 2:
                    shutil.copy2(src_file, dest_file)
                    synced += 1
        except Exception as e:
            print(f"Warning: could not sync {src_file} -> {dest_file}: {e}", flush=True)

    if synced > 0:
        elapsed = time.time() - start
        print(
            f"[SYNC] Synced {synced} {label} files from host to fast volume in {elapsed:.2f}s",
            flush=True,
        )


def main():
    print("=" * 60, flush=True)
    print("SKU MatchOps ML Engine Container starting...", flush=True)
    print("=" * 60, flush=True)

    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")

    def parse_url(url, default_port=80):
        parts = (
            url.replace("http://", "").replace("https://", "").replace("redis://", "").split("/")
        )
        host_port = parts[0].split("@")[-1].split(":")
        host = host_port[0]
        port = int(host_port[1]) if len(host_port) > 1 else default_port
        return host, port

    try:
        q_host, q_port = parse_url(qdrant_url, default_port=6333)
        if not wait_for_service(q_host, q_port, timeout=45.0):
            sys.exit(1)
    except Exception as e:
        print(f"FATAL: Could not connect to QDRANT_URL={qdrant_url}: {e}", flush=True)
        sys.exit(1)

    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    try:
        r_host, r_port = parse_url(redis_url, default_port=6379)
        if not wait_for_service(r_host, r_port, timeout=30.0):
            sys.exit(1)
    except Exception as e:
        print(f"FATAL: Could not connect to REDIS_URL={redis_url}: {e}", flush=True)
        sys.exit(1)

    # Prune stale volume caches if host active_embedding_model.txt changed
    host_marker = Path("/host_cache/active_embedding_model.txt")
    vol_marker = Path("/app/data/cache/active_embedding_model.txt")
    if host_marker.exists() and vol_marker.exists():
        if (
            host_marker.read_text(encoding="utf-8").strip()
            != vol_marker.read_text(encoding="utf-8").strip()
        ):
            print(
                "[SYNC] Host active_embedding_model.txt changed; clearing stale volume cache...",
                flush=True,
            )
            for f in Path("/app/data/cache").iterdir():
                if f.is_file():
                    f.unlink(missing_ok=True)

    # Sync from host bind-mounts to fast Docker volumes if present
    sync_host_directory("/host_models", "/app/data/models", "model")
    sync_host_directory("/host_cache", "/app/data/cache", "cache")

    # Run synchronous pre-flight startup checks (fails fast on any mismatch or missing artifact)
    try:
        subprocess.run(
            [sys.executable, "-m", "engine.pipelines.startup_check_pipeline"], check=True
        )
    except subprocess.CalledProcessError:
        print("FATAL: Engine pre-flight startup checks failed. Aborting startup.", flush=True)
        sys.exit(1)

    # Launch the gRPC Engine microservice on ENGINE_GRPC_PORT (50051)
    grpc_port = os.getenv("ENGINE_GRPC_PORT", "50051")
    print("=" * 60, flush=True)
    print(f"Starting SKU MatchOps ML Engine gRPC server on 0.0.0.0:{grpc_port}...", flush=True)
    print("=" * 60, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()

    os.execvp(sys.executable, [sys.executable, "-m", "engine.server"])


if __name__ == "__main__":
    main()
