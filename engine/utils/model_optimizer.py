import os
import numpy as np
import onnxruntime as ort
from sentence_transformers import CrossEncoder
import torch
from transformers import AutoModel, AutoTokenizer

from engine import config

def export_bge_m3_onnx(model_name: str, export_path: str):
    """Reference exporter for standard single-head BGE-M3."""
    print(f"Exporting {model_name} to ONNX...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    dummy_input = tokenizer("Optimizing SKU MatchOps", return_tensors="pt")

    symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
    torch.onnx.export(
        model,
        (dummy_input['input_ids'], dummy_input['attention_mask']),
        export_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['last_hidden_state'],
        dynamic_axes={
            'input_ids': symbolic_names,
            'attention_mask': symbolic_names,
            'last_hidden_state': symbolic_names
        },
        opset_version=14
    )
    print(f"Done: {export_path}")

def export_cross_encoder_onnx(model_name: str, export_path: str):
    """Reference exporter for standard cross-encoder."""
    print(f"Exporting Cross-Encoder {model_name} to ONNX...")
    model = CrossEncoder(model_name, device='cpu')
    tokenizer = model.tokenizer
    hf_model = model.model
    hf_model.eval()

    dummy_input = tokenizer("Query", "Document", return_tensors="pt")

    symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
    torch.onnx.export(
        hf_model,
        (dummy_input['input_ids'], dummy_input['attention_mask']),
        export_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': symbolic_names,
            'attention_mask': symbolic_names,
            'logits': {0: 'batch_size'}
        },
        opset_version=14
    )
    print(f"Done: {export_path}")

def get_onnx_session(model_path: str) -> ort.InferenceSession:
    """Helper to initialize an optimized ONNX CPU inference session."""
    sess_options = ort.SessionOptions()
    sess_options.add_session_config_entry("session.use_mmap_for_weights", "1")
    sess_options.intra_op_num_threads = config.MAX_CPU_CORES
    sess_options.inter_op_num_threads = config.MAX_CPU_CORES
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
    return session

def warmup_onnx(session, batch_size=1, seq_len=32):
    inputs = {}
    for input_meta in session.get_inputs():
        name = input_meta.name
        shape = [batch_size, seq_len] if 'logits' not in name else [batch_size] # Simplistic check
        # Fix shape for dynamic axes
        actual_shape = []
        for s in input_meta.shape:
            if isinstance(s, str) or s is None or s <= 0:
                if 'batch' in input_meta.name or (isinstance(s, str) and 'batch' in s):
                    actual_shape.append(batch_size)
                else:
                    actual_shape.append(seq_len)
            else:
                actual_shape.append(s)

        inputs[name] = np.zeros(actual_shape, dtype=np.int64)

    session.run(None, inputs)
    print(f"Warmup complete for {os.path.basename(session._model_path if hasattr(session, '_model_path') else 'model')}")
