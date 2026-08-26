import os
import sys
import subprocess
import time
import socket

def wait_for_service(host: str, port: int, timeout: float = 60.0):
    """Waits for a service to become available on host:port."""
    start_time = time.time()
    print(f"Waiting for database service {host}:{port}...", flush=True)
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

    # Check and prepare ONNX & INT8 models (Idempotent)
    sys.path.insert(0, "/app")
    try:
        from engine import config
        
        bge_int8 = os.path.join(config.ONNX_DIR, "bge_m3", "model_int8.onnx")
        bge_fp32 = os.path.join(config.ONNX_DIR, "bge_m3", "model.onnx")
        rerank_int8 = os.path.join(config.ONNX_DIR, "reranker", "model_int8.onnx")
        rerank_fp32 = os.path.join(config.ONNX_DIR, "reranker", "model.onnx")
        gliner_onnx = os.path.join(config.ONNX_DIR, "gliner", "model.onnx")
        
        use_int8 = config.USE_INT8_MODELS
        
        needs_export = False
        needs_quantization = False
        
        # Check BGE-M3
        if use_int8:
            if not os.path.exists(bge_int8):
                needs_quantization = True
                if not os.path.exists(bge_fp32):
                    needs_export = True
        else:
            if not os.path.exists(bge_fp32):
                needs_export = True

        # Check Reranker
        if use_int8:
            if not os.path.exists(rerank_int8):
                needs_quantization = True
                if not os.path.exists(rerank_fp32):
                    needs_export = True
        else:
            if not os.path.exists(rerank_fp32):
                needs_export = True

        # Check GLiNER
        if not os.path.exists(gliner_onnx):
            needs_export = True

        if needs_export:
            print("[SETUP] Required base ONNX models are missing. Initiating one-time export/download...", flush=True)
            env = os.environ.copy()
            env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            subprocess.run([sys.executable, "engine/export_onnx.py"], check=True, env=env)
            print("[SETUP] Base ONNX models exported successfully!", flush=True)

        if needs_quantization and use_int8:
            print("[SETUP] Running INT8 dynamic quantization for memory optimization...", flush=True)
            subprocess.run([sys.executable, "engine/quantize_models.py"], check=True)
            print("[SETUP] INT8 quantization complete!", flush=True)

        if not needs_export and not needs_quantization:
            print("[OK] All required ONNX INT8 models and configs are present (cached).", flush=True)

    except Exception as e:
        print(f"Error checking/preparing ONNX models: {e}", flush=True)
        print("Attempting to proceed with server startup...", flush=True)

    # Launch uvicorn for the Engine microservice on port 8001
    port = os.getenv("ENGINE_PORT", "8001")
    print("=" * 60, flush=True)
    print(f"Starting SKU MatchOps ML Engine on 0.0.0.0:{port}...", flush=True)
    print("=" * 60, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    
    os.execvp("uvicorn", ["uvicorn", "engine.server:app", "--host", "0.0.0.0", "--port", port])

if __name__ == "__main__":
    main()
