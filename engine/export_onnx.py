import logging
import os
import shutil
import sys

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("export_onnx")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_DIR = os.path.join(BASE_DIR, "onnx_models")
TMP_DIR = os.path.join(BASE_DIR, "onnx_tmp")

def ensure_dirs():
    os.makedirs(ONNX_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

class BGEHybridONNXWrapper(nn.Module):
    def __init__(self, flag_model):
        super().__init__()
        self.encoder = flag_model.model.model
        self.sparse_linear = flag_model.model.sparse_linear
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden_states = outputs.last_hidden_state
        dense_vecs = hidden_states[:, 0, :]
        sparse_weights = self.sparse_linear(hidden_states)
        sparse_weights = torch.relu(sparse_weights)
        sparse_weights = sparse_weights.squeeze(-1)
        return dense_vecs, sparse_weights

def export_bge_m3(force: bool = False, quantize: bool = True):
    target_dir = os.path.join(ONNX_DIR, "bge_m3")
    onnx_fp32 = os.path.join(target_dir, "model.onnx")
    onnx_int8 = os.path.join(target_dir, "model_int8.onnx")
    tokenizer_file = os.path.join(target_dir, "tokenizer.json")
    
    if not force and (os.path.exists(onnx_int8) or os.path.exists(onnx_fp32)) and os.path.exists(tokenizer_file):
        logger.info("BGE-M3 ONNX model already exists. Skipping export.")
        if quantize and not os.path.exists(onnx_int8) and os.path.exists(onnx_fp32):
            try:
                from engine.quantize_models import quantize_model
                quantize_model(onnx_fp32, onnx_int8, "BGE-M3")
                if os.getenv("CLEANUP_FP32_MODELS", "true").lower() == "true" and os.path.exists(onnx_int8):
                    os.remove(onnx_fp32)
            except Exception as q_err:
                logger.warning(f"Failed to quantize existing BGE-M3: {q_err}")
        return

    logger.info("Exporting Hybrid BGE-M3 (Dense + Sparse) to ONNX...")
    try:
        import gc
        from transformers import AutoTokenizer
        from FlagEmbedding import BGEM3FlagModel
        
        os.makedirs(target_dir, exist_ok=True)

        logger.info("Loading BGEM3FlagModel ('BAAI/bge-m3')...")
        flag_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)
        wrapper = BGEHybridONNXWrapper(flag_model)
        wrapper.eval()

        tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
        inputs = tokenizer(["Test"], padding=True, truncation=True, return_tensors="pt")

        logger.info("Tracing and exporting via torch.onnx.export...")
        torch.onnx.export(
            wrapper,
            (inputs["input_ids"], inputs["attention_mask"]),
            onnx_fp32,
            input_names=['input_ids', 'attention_mask'],
            output_names=['dense_vecs', 'sparse_weights'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'dense_vecs': {0: 'batch_size'},
                'sparse_weights': {0: 'batch_size', 1: 'sequence_length'}
            },
            opset_version=14,
            do_constant_folding=True,
        )

        tokenizer.save_pretrained(target_dir)
        del flag_model
        del wrapper
        del tokenizer
        del inputs
        gc.collect()
        logger.info(f"BGE-M3 Hybrid exported successfully to {onnx_fp32}")

        if quantize and (os.getenv("USE_INT8_MODELS", "true").lower() == "true"):
            from engine.quantize_models import quantize_model
            quantize_model(onnx_fp32, onnx_int8, "BGE-M3")
            if os.getenv("CLEANUP_FP32_MODELS", "true").lower() == "true" and os.path.exists(onnx_int8):
                try:
                    os.remove(onnx_fp32)
                    logger.info("[BGE-M3] Deleted unquantized FP32 model to reclaim disk.")
                except Exception as e:
                    logger.warning(f"Could not remove FP32 model {onnx_fp32}: {e}")
    except Exception as e:
        logger.error(f"Failed to export Hybrid BGE-M3: {e}")
        raise

def export_bge_reranker(force: bool = False, quantize: bool = True):
    target_dir = os.path.join(ONNX_DIR, "reranker")
    onnx_fp32 = os.path.join(target_dir, "model.onnx")
    onnx_int8 = os.path.join(target_dir, "model_int8.onnx")
    tokenizer_file = os.path.join(target_dir, "tokenizer.json")

    if not force and (os.path.exists(onnx_int8) or os.path.exists(onnx_fp32)) and os.path.exists(tokenizer_file):
        logger.info("BGE-Reranker ONNX model already exists. Skipping export.")
        if quantize and not os.path.exists(onnx_int8) and os.path.exists(onnx_fp32):
            try:
                from engine.quantize_models import quantize_model
                quantize_model(onnx_fp32, onnx_int8, "BGE-RERANKER")
                if os.getenv("CLEANUP_FP32_MODELS", "true").lower() == "true" and os.path.exists(onnx_int8):
                    os.remove(onnx_fp32)
            except Exception as q_err:
                logger.warning(f"Failed to quantize existing BGE-Reranker: {q_err}")
        return

    logger.info("Exporting BGE-Reranker-v2-M3 Cross-Encoder to ONNX...")
    try:
        import gc
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        os.makedirs(target_dir, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
        model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3")
        model.eval()

        inputs = tokenizer(["query"], ["passage"], padding=True, truncation=True, return_tensors="pt")

        logger.info("Tracing and exporting BGE-Reranker via torch.onnx.export...")
        torch.onnx.export(
            model,
            (inputs["input_ids"], inputs["attention_mask"]),
            onnx_fp32,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size'}
            },
            opset_version=14,
            do_constant_folding=True,
        )

        tokenizer.save_pretrained(target_dir)
        del model
        del tokenizer
        del inputs
        gc.collect()
        logger.info(f"BGE-Reranker exported successfully to {onnx_fp32}")

        if quantize and (os.getenv("USE_INT8_MODELS", "true").lower() == "true"):
            from engine.quantize_models import quantize_model
            quantize_model(onnx_fp32, onnx_int8, "BGE-RERANKER")
            if os.getenv("CLEANUP_FP32_MODELS", "true").lower() == "true" and os.path.exists(onnx_int8):
                try:
                    os.remove(onnx_fp32)
                    logger.info("[BGE-RERANKER] Deleted unquantized FP32 model to reclaim disk.")
                except Exception as e:
                    logger.warning(f"Could not remove FP32 model {onnx_fp32}: {e}")
    except Exception as e:
        logger.error(f"Failed to export BGE-Reranker: {e}")
        raise

def export_gliner(force: bool = False):
    target_dir = os.path.join(ONNX_DIR, "gliner")
    onnx_file = os.path.join(target_dir, "model.onnx")
    config_file = os.path.join(target_dir, "gliner_config.json")

    if not force and os.path.exists(onnx_file) and os.path.exists(config_file):
        logger.info("GLiNER ONNX model already exists. Skipping export.")
        return

    logger.info("Downloading and exporting GLiNER Medium v2.1 ONNX model...")
    try:
        import gc
        from huggingface_hub import snapshot_download
        os.makedirs(target_dir, exist_ok=True)
        
        # Download the ONNX converted repository for GLiNER
        snapshot_download(
            repo_id="onnx-community/gliner_medium-v2.1",
            local_dir=target_dir,
            allow_patterns=["*.json", "*.txt", "model.onnx", "onnx/model.onnx"]
        )
        
        # Move the onnx model from the subfolder if present
        nested_onnx = os.path.join(target_dir, "onnx", "model.onnx")
        if os.path.exists(nested_onnx):
            shutil.move(nested_onnx, onnx_file)
            shutil.rmtree(os.path.join(target_dir, "onnx"), ignore_errors=True)
            
        gc.collect()
        logger.info(f"Downloaded full GLiNER ONNX model successfully to {target_dir}")
    except Exception as e:
        logger.error(f"Failed to process GLiNER ONNX: {e}")
        raise

def cleanup():
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR, ignore_errors=True)

def export_all_models_if_needed(force: bool = False, quantize: bool = True):
    ensure_dirs()
    export_bge_m3(force=force, quantize=quantize)
    export_bge_reranker(force=force, quantize=quantize)
    export_gliner(force=force)
    cleanup()
    logger.info("ONNX model preparation complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SKU MatchOps ONNX Exporter & Downloader")
    parser.add_argument("--force", action="store_true", help="Force re-export even if models exist")
    parser.add_argument("--no-quantize", action="store_true", help="Skip INT8 dynamic quantization after export")
    args = parser.parse_args()

    logger.info("Starting ONNX Model Export for MatchOps...")
    export_all_models_if_needed(force=args.force, quantize=not args.no_quantize)
