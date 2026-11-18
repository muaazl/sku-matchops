import os
import shutil
import socket
import subprocess
import sys
import time


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
                    f"WARNING: Timeout waiting for service {host}:{port}. Attempting to proceed anyway...",
                    flush=True,
                )
                return False
            time.sleep(1.0)


def sync_host_directory(host_dir: str, target_dir: str, label: str):
    """Syncs missing or updated files from host bind-mount into fast Docker volume."""
    if not os.path.exists(host_dir):
        return
    synced = 0
    start = time.time()
    for root, _dirs, files in os.walk(host_dir):
        rel = os.path.relpath(root, host_dir)
        dest_root = target_dir if rel == "." else os.path.join(target_dir, rel)
        os.makedirs(dest_root, exist_ok=True)
        for f in files:
            src_file = os.path.join(root, f)
            dest_file = os.path.join(dest_root, f)
            try:
                s_stat = os.stat(src_file)
                if not os.path.exists(dest_file):
                    shutil.copy2(src_file, dest_file)
                    synced += 1
                else:
                    d_stat = os.stat(dest_file)
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

    def parse_url(url):
        parts = url.replace("http://", "").replace("https://", "").split("/")
        host_port = parts[0].split(":")
        host = host_port[0]
        port = int(host_port[1]) if len(host_port) > 1 else 80
        return host, port

    try:
        q_host, q_port = parse_url(qdrant_url)
        wait_for_service(q_host, q_port, timeout=45.0)
    except Exception as e:
        print(f"Could not parse or connect to QDRANT_URL={qdrant_url}: {e}", flush=True)

    # Sync from host bind-mounts to fast Docker volumes if present
    sync_host_directory("/host_models", "/app/engine/models", "model")
    sync_host_directory("/host_cache", "/app/engine/data/cache", "cache")

    # Check and prepare ONNX & INT8 models (Idempotent)
    sys.path.insert(0, "/app")
    try:
        from engine import config

        emb_int8 = config.EMBEDDING_ONNX_INT8
        emb_fp32 = config.EMBEDDING_ONNX_FP32
        rerank_int8 = config.RERANKER_ONNX_INT8
        rerank_fp32 = config.RERANKER_ONNX_FP32
        gliner_onnx = config.GLINER_ONNX

        needs_export = False
        needs_quantization = False

        # Check active embedding model (strictly INT8 ONNX)
        if not os.path.exists(emb_int8):
            if os.path.exists(emb_fp32):
                needs_quantization = True
            else:
                needs_export = True

        # Check active reranker model (strictly INT8 ONNX)
        if not os.path.exists(rerank_int8):
            if os.path.exists(rerank_fp32):
                needs_quantization = True
            else:
                needs_export = True

        # Check GLiNER ONNX
        if not os.path.exists(gliner_onnx):
            needs_export = True

        if needs_export:
            print(
                "[SETUP] Required models are missing. Initiating download, ONNX export & INT8 quantization...",
                flush=True,
            )
            env = os.environ.copy()
            env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            subprocess.run(
                [sys.executable, "-m", "scripts.sync", "--models", "sync"],
                check=True,
                env=env,
            )
            print("[SETUP] Models exported and quantized successfully!", flush=True)

        if needs_quantization and (not os.path.exists(emb_int8) or not os.path.exists(rerank_int8)):
            print(
                "[SETUP] Running INT8 dynamic quantization for memory optimization...", flush=True
            )
            subprocess.run(
                [sys.executable, "-m", "scripts.sync", "--models", "quantize"],
                check=True,
            )
            print("[SETUP] INT8 quantization complete!", flush=True)

        if not needs_export and not needs_quantization:
            print("[OK] All required ONNX models and configs are present (cached).", flush=True)

        for required_path in (emb_int8, rerank_int8, gliner_onnx):
            if not os.path.exists(required_path):
                raise FileNotFoundError(
                    f"Required ONNX artifact still missing after setup: {required_path}"
                )

    except Exception as e:
        print(f"FATAL: ONNX model setup failed ({e})", flush=True)
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
