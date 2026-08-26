import argparse
import os
import sys
import time

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers import AutoTokenizer

# Resolve ONNX models directory dynamically
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ONNX_DIR = os.path.join(PROJECT_ROOT, "engine", "onnx_models")

def quantize_model(input_path: str, output_path: str, model_name: str):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Source FP32 model not found at: {input_path}")

    print(f"[{model_name}] Starting INT8 dynamic quantization from {input_path}...")
    start_t = time.time()
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
        extra_options={"MatMulConstBOnly": True},
    )
    elapsed = time.time() - start_t
    in_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    
    # Check for external data files if present
    for suffix in ["_data", ".data"]:
        data_path = input_path + suffix
        if os.path.exists(data_path):
            in_size_mb += os.path.getsize(data_path) / (1024 * 1024)
        
    out_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    for suffix in ["_data", ".data"]:
        out_data = output_path + suffix
        if os.path.exists(out_data):
            out_size_mb += os.path.getsize(out_data) / (1024 * 1024)
        
    reduction = (1 - (out_size_mb / in_size_mb)) * 100 if in_size_mb > 0 else 0
    print(f"[{model_name}] Quantization completed in {elapsed:.2f}s!")
    print(f"[{model_name}] Original: {in_size_mb:.1f} MB -> Quantized: {out_size_mb:.1f} MB ({reduction:.1f}% reduction)")

def verify_bge_m3():
    bge_dir = os.path.join(ONNX_DIR, "bge_m3")
    fp32_model = os.path.join(bge_dir, "model.onnx")
    int8_model = os.path.join(bge_dir, "model_int8.onnx")

    if not (os.path.exists(fp32_model) and os.path.exists(int8_model)):
        print("[VERIFY] Skipping BGE-M3 comparison (either FP32 or INT8 model file is missing).")
        return

    print("\n[VERIFY] Testing BGE-M3 INT8 vs FP32...")
    tokenizer = AutoTokenizer.from_pretrained(bge_dir, local_files_only=True)
    sample_texts = ["Red Bull Energy Drink 250ml", "Coca Cola Zero Sugar 330ml Can", "Fresh organic bananas 1kg"]
    inputs = tokenizer(sample_texts, padding=True, truncation=True, return_tensors="np")
    ort_inputs = {k: v.astype(np.int64) for k, v in inputs.items()}

    t0 = time.time()
    fp32_sess = ort.InferenceSession(fp32_model, providers=['CPUExecutionProvider'])
    fp32_load_t = time.time() - t0
    t0 = time.time()
    fp32_out = fp32_sess.run(None, ort_inputs)[0]
    fp32_infer_t = time.time() - t0

    t0 = time.time()
    int8_sess = ort.InferenceSession(int8_model, providers=['CPUExecutionProvider'])
    int8_load_t = time.time() - t0
    t0 = time.time()
    int8_out = int8_sess.run(None, ort_inputs)[0]
    int8_infer_t = time.time() - t0

    fp32_norm = fp32_out / np.linalg.norm(fp32_out, axis=1, keepdims=True)
    int8_norm = int8_out / np.linalg.norm(int8_out, axis=1, keepdims=True)
    cos_sims = np.sum(fp32_norm * int8_norm, axis=1)

    print(f"BGE-M3 Load Time:  FP32 = {fp32_load_t:.2f}s | INT8 = {int8_load_t:.2f}s ({fp32_load_t/max(int8_load_t, 1e-4):.1f}x speedup)")
    print(f"BGE-M3 Infer Time: FP32 = {fp32_infer_t*1000:.1f}ms | INT8 = {int8_infer_t*1000:.1f}ms ({fp32_infer_t/max(int8_infer_t, 1e-4):.1f}x speedup)")
    for text, sim in zip(sample_texts, cos_sims):
        print(f"  - '{text}': Cosine Similarity = {sim:.5f} ({sim*100:.2f}% match with FP32)")

def verify_reranker():
    rerank_dir = os.path.join(ONNX_DIR, "reranker")
    fp32_model = os.path.join(rerank_dir, "model.onnx")
    int8_model = os.path.join(rerank_dir, "model_int8.onnx")

    if not (os.path.exists(fp32_model) and os.path.exists(int8_model)):
        print("[VERIFY] Skipping BGE-Reranker comparison (either FP32 or INT8 model file is missing).")
        return

    print("\n[VERIFY] Testing BGE-Reranker INT8 vs FP32...")
    tokenizer = AutoTokenizer.from_pretrained(rerank_dir, local_files_only=True)
    pairs = [
        ("Red Bull Energy Drink", "Red Bull Energy Drink 250ml Can"),
        ("Coca Cola Zero", "Pepsi Max 330ml Can"),
    ]
    inputs = tokenizer([p[0] for p in pairs], [p[1] for p in pairs], padding=True, truncation=True, return_tensors="np")
    ort_inputs = {k: v.astype(np.int64) for k, v in inputs.items()}

    t0 = time.time()
    fp32_sess = ort.InferenceSession(fp32_model, providers=['CPUExecutionProvider'])
    fp32_load_t = time.time() - t0
    fp32_logits = fp32_sess.run(None, ort_inputs)[0]

    t0 = time.time()
    int8_sess = ort.InferenceSession(int8_model, providers=['CPUExecutionProvider'])
    int8_load_t = time.time() - t0
    int8_logits = int8_sess.run(None, ort_inputs)[0]

    print(f"Reranker Load Time: FP32 = {fp32_load_t:.2f}s | INT8 = {int8_load_t:.2f}s ({fp32_load_t/max(int8_load_t, 1e-4):.1f}x speedup)")
    for (q, doc), f_val, i_val in zip(pairs, fp32_logits.flatten(), int8_logits.flatten()):
        print(f"  - ({q} <-> {doc}): FP32={f_val:.4f}, INT8={i_val:.4f} (diff = {abs(f_val - i_val):.4f})")

def quantize_all_if_needed(verify: bool = False, force: bool = False):
    bge_in = os.path.join(ONNX_DIR, "bge_m3", "model.onnx")
    bge_out = os.path.join(ONNX_DIR, "bge_m3", "model_int8.onnx")
    
    if force or not os.path.exists(bge_out):
        if os.path.exists(bge_in):
            quantize_model(bge_in, bge_out, "BGE-M3")
        else:
            print(f"[WARN] Cannot quantize BGE-M3: source {bge_in} does not exist.")
    else:
        print("[OK] BGE-M3 INT8 model already exists. Skipping quantization.")

    rerank_in = os.path.join(ONNX_DIR, "reranker", "model.onnx")
    rerank_out = os.path.join(ONNX_DIR, "reranker", "model_int8.onnx")
    
    if force or not os.path.exists(rerank_out):
        if os.path.exists(rerank_in):
            quantize_model(rerank_in, rerank_out, "BGE-RERANKER")
        else:
            print(f"[WARN] Cannot quantize BGE-Reranker: source {rerank_in} does not exist.")
    else:
        print("[OK] BGE-Reranker INT8 model already exists. Skipping quantization.")

    if verify:
        verify_bge_m3()
        verify_reranker()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SKU MatchOps INT8 Dynamic Quantizer")
    parser.add_argument("--verify", action="store_true", help="Run FP32 vs INT8 verification benchmarks")
    parser.add_argument("--force", action="store_true", help="Force re-quantization even if output exists")
    args = parser.parse_args()

    quantize_all_if_needed(verify=args.verify, force=args.force)
