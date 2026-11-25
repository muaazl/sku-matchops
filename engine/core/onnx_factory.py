"""
Shared ONNX Runtime session factory and process memory helpers for Engine models.
"""

from __future__ import annotations

import ctypes
import gc
from pathlib import Path
from typing import Any


def check_cuda_provider_available() -> bool:
    """Returns True when onnxruntime reports CUDAExecutionProvider as available."""
    try:
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def resolve_onnx_providers(
    onnx_path: Path | str,
    default_providers: list[str] | None = None,
) -> tuple[list[Any], bool]:
    """Resolves ORT execution providers and whether CUDA acceleration is enabled."""
    if str(onnx_path).endswith("_int8.onnx"):
        return ["CPUExecutionProvider"], False
    providers = list(default_providers or ["CPUExecutionProvider"])
    uses_cuda = "CUDAExecutionProvider" in providers
    if uses_cuda:
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:
                pass
    formatted: list[Any] = [
        ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})
        if p == "CUDAExecutionProvider"
        else p
        for p in providers
    ]
    return formatted, uses_cuda


def build_onnx_session_options(max_cpu_cores: int, uses_cuda: bool = False) -> Any:
    """Builds ONNX Runtime SessionOptions with mmap weight loading and bounded CPU threads."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = False
    if not uses_cuda:
        opts.add_session_config_entry("session.use_mmap_for_weights", "1")
        opts.intra_op_num_threads = max(1, int(max_cpu_cores))
        opts.inter_op_num_threads = 1
    return opts


def create_onnx_session(
    onnx_path: Path | str,
    default_providers: list[str] | None = None,
    max_cpu_cores: int = 1,
) -> tuple[Any, bool]:
    """Creates an ONNX Runtime InferenceSession and returns (session, uses_cuda)."""
    import onnxruntime as ort

    providers, uses_cuda = resolve_onnx_providers(onnx_path, default_providers)
    sess_opts = build_onnx_session_options(max_cpu_cores=max_cpu_cores, uses_cuda=uses_cuda)
    session = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=providers)
    return session, uses_cuda


def filter_onnx_inputs(session: Any, ort_inputs: dict[str, Any]) -> dict[str, Any]:
    """Filters input tensors to those declared by the ONNX model graph."""
    if hasattr(session, "get_inputs"):
        try:
            valid_names = {
                inp.name
                for inp in session.get_inputs()
                if isinstance(getattr(inp, "name", None), str)
            }
            if valid_names:
                return {k: v for k, v in ort_inputs.items() if k in valid_names}
        except Exception:
            pass
    return ort_inputs


def trim_process_memory() -> None:
    """Runs Python GC and returns freed glibc heap arenas to the OS on Linux."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
