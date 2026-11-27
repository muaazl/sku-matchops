"""Architecture layer and dependency boundary enforcement tests for engine/."""

from __future__ import annotations

import ast
from pathlib import Path

ENGINE_ROOT = Path(__file__).resolve().parents[1] / "engine"
BACKEND_ROOT = Path(__file__).resolve().parents[1] / "backend"

APPROVED_ENGINE_SUBPACKAGES = {
    "core",
    "preprocessing",
    "vector_store",
    "inference",
    "pipelines",
    "grpc",
    "protos",
}

APPROVED_ENGINE_ROOT_FILES = {"__init__.py", "server.py", "requirements.txt"}

# Strictly enforce: grpc -> pipelines -> (inference & vector_store) -> preprocessing -> core (+ protos)
FORBIDDEN_IMPORTS_BY_LAYER: dict[str, tuple[str, ...]] = {
    "core": (
        "engine.preprocessing",
        "engine.vector_store",
        "engine.inference",
        "engine.pipelines",
        "engine.grpc",
        "engine.protos",
        "grpc",
        "qdrant_client",
    ),
    "preprocessing": (
        "engine.vector_store",
        "engine.inference",
        "engine.pipelines",
        "engine.grpc",
        "engine.protos",
        "grpc",
        "qdrant_client",
    ),
    "vector_store": (
        "engine.inference",
        "engine.pipelines",
        "engine.grpc",
        "engine.protos",
        "grpc",
    ),
    "inference": (
        "engine.vector_store",
        "engine.pipelines",
        "engine.grpc",
        "engine.protos",
        "grpc",
        "qdrant_client",
    ),
    "pipelines": (
        "engine.grpc",
        "engine.protos",
        "grpc",
        "qdrant_client",
    ),
    "grpc": (
        "engine.vector_store",
        "engine.inference",
        "qdrant_client",
    ),
}


def _imported_modules(py_file: Path) -> list[str]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
    return modules


def test_engine_directory_contains_only_approved_layers_and_root_files():
    dirs = {p.name for p in ENGINE_ROOT.iterdir() if p.is_dir() and p.name != "__pycache__"}
    files = {p.name for p in ENGINE_ROOT.iterdir() if p.is_file()}

    assert dirs == APPROVED_ENGINE_SUBPACKAGES
    assert files == APPROVED_ENGINE_ROOT_FILES


def test_engine_layer_import_dag_and_isolation_rules():
    violations: list[tuple[str, str]] = []

    for layer, forbidden_prefixes in FORBIDDEN_IMPORTS_BY_LAYER.items():
        layer_dir = ENGINE_ROOT / layer
        assert layer_dir.is_dir(), f"Missing expected engine layer '{layer}'"
        for py_file in layer_dir.rglob("*.py"):
            if "__pycache__" in py_file.parts:
                continue
            for mod in _imported_modules(py_file):
                if any(mod == p or mod.startswith(f"{p}.") for p in forbidden_prefixes):
                    violations.append((py_file.relative_to(ENGINE_ROOT).as_posix(), mod))

    assert not violations, f"Layer boundary violations found: {violations}"


def test_queue_pipeline_has_no_circular_import_to_worker_pipeline():
    queue_file = ENGINE_ROOT / "pipelines" / "queue_pipeline.py"
    mods = _imported_modules(queue_file)
    assert not any("worker_pipeline" in m for m in mods)


def test_no_persistent_artifacts_stored_inside_code_packages():
    forbidden_suffixes = {".pkl", ".joblib", ".feather", ".csv", ".onnx"}
    offending: list[str] = []
    for root in (ENGINE_ROOT, BACKEND_ROOT):
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in forbidden_suffixes:
                offending.append(path.as_posix())
    assert not offending, f"Data/model files must live under ./data/: {offending}"
