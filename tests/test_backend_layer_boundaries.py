"""Architectural boundary and naming tests for `backend/` package layers."""

import ast
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1] / "backend"

APPROVED_BACKEND_ROOT_FILES = {"__init__.py", "main.py", "requirements.txt"}
APPROVED_BACKEND_SUBPACKAGES = {
    "api",
    "core",
    "grpc",
    "models",
    "protos",
    "repositories",
    "schemas",
    "services",
}

FORBIDDEN_IMPORTS_BY_LAYER: dict[str, set[str]] = {
    "core": {
        "backend.models",
        "backend.schemas",
        "backend.repositories",
        "backend.services",
        "backend.grpc",
        "backend.protos",
        "backend.api",
        "grpc",
        "redis",
        "engine",
    },
    "models": {
        "backend.core",
        "backend.schemas",
        "backend.repositories",
        "backend.services",
        "backend.grpc",
        "backend.protos",
        "backend.api",
        "grpc",
        "redis",
        "fastapi",
        "starlette",
        "pydantic",
        "engine",
    },
    "schemas": {
        "backend.models",
        "backend.repositories",
        "backend.services",
        "backend.grpc",
        "backend.protos",
        "backend.api",
        "sqlalchemy",
        "grpc",
        "redis",
        "fastapi",
        "starlette",
        "engine",
    },
    "repositories": {
        "backend.services",
        "backend.grpc",
        "backend.protos",
        "backend.api",
        "grpc",
        "fastapi",
        "starlette",
        "engine",
    },
    "services": {
        "backend.models",
        "backend.grpc",
        "backend.protos",
        "backend.api",
        "sqlalchemy",
        "redis",
        "grpc",
        "fastapi",
        "starlette",
        "engine",
    },
    "protos": {
        "backend.core",
        "backend.models",
        "backend.schemas",
        "backend.repositories",
        "backend.services",
        "backend.grpc",
        "backend.api",
        "sqlalchemy",
        "redis",
        "fastapi",
        "starlette",
        "engine",
    },
    "grpc": {
        "backend.api",
        "backend.repositories",
        "backend.models",
        "sqlalchemy",
        "redis",
        "fastapi",
        "starlette",
        "engine",
    },
    "api": {
        "backend.repositories",
        "backend.models",
        "backend.grpc",
        "backend.protos",
        "sqlalchemy",
        "redis",
        "grpc",
        "engine",
    },
    "main": {
        "backend.repositories",
        "backend.models",
        "backend.protos",
        "sqlalchemy",
        "redis",
        "grpc",
        "engine",
    },
}

_REQUIRED_SUFFIXES = {
    ("models",): "_model.py",
    ("schemas",): "_schema.py",
    ("repositories",): "_repository.py",
    ("services",): "_service.py",
    ("api", "endpoints"): "_router.py",
    ("api", "middleware"): "_middleware.py",
}


def _imported_modules(py_file: Path) -> set[str]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    return mods


def _matches_forbidden(imported: str, forbidden: set[str]) -> bool:
    return any(imported == f or imported.startswith(f + ".") for f in forbidden)


def test_backend_root_structure_only_approved_packages():
    """No spurious `app/` package or ad-hoc root files may exist in `backend/`."""
    assert not (BACKEND_ROOT / "app").exists()
    root_dirs = {p.name for p in BACKEND_ROOT.iterdir() if p.is_dir() and p.name != "__pycache__"}
    root_files = {p.name for p in BACKEND_ROOT.iterdir() if p.is_file()}
    assert root_dirs == APPROVED_BACKEND_SUBPACKAGES
    assert root_files == APPROVED_BACKEND_ROOT_FILES


def test_backend_layer_file_naming_suffixes():
    """All layer module filenames follow their role suffix conventions."""
    for rel_parts, expected_suffix in _REQUIRED_SUFFIXES.items():
        directory = BACKEND_ROOT.joinpath(*rel_parts)
        assert directory.is_dir()
        for py_file in directory.glob("*.py"):
            if py_file.name == "__init__.py":
                continue
            assert py_file.name.endswith(expected_suffix), (
                f"{py_file} must end with {expected_suffix}"
            )


def test_backend_layer_imports_respect_strict_dag():
    """Every module under `backend/` imports only permitted downward/external packages."""
    violations: list[str] = []
    for py_file in sorted(BACKEND_ROOT.rglob("*.py")):
        rel = py_file.relative_to(BACKEND_ROOT)
        layer = "main" if len(rel.parts) == 1 else rel.parts[0]
        if layer == "protos" and "_pb2" in py_file.name:
            # Skip static grpc/protobuf checks inside generated stubs beyond backend.* check
            forbidden = FORBIDDEN_IMPORTS_BY_LAYER["protos"]
        else:
            forbidden = FORBIDDEN_IMPORTS_BY_LAYER[layer]

        imported = _imported_modules(py_file)
        for mod in sorted(imported):
            if _matches_forbidden(mod, forbidden):
                violations.append(f"backend/{rel.as_posix()} imports forbidden '{mod}'")
            if (
                layer == "core"
                and rel.name != "database.py"
                and (mod == "sqlalchemy" or mod.startswith("sqlalchemy."))
            ):
                violations.append(
                    f"backend/{rel.as_posix()} imports sqlalchemy outside core/database.py"
                )
            if (
                layer == "core"
                and rel.name != "security.py"
                and (
                    mod == "fastapi"
                    or mod.startswith("fastapi.")
                    or mod == "starlette"
                    or mod.startswith("starlette.")
                )
            ):
                violations.append(
                    f"backend/{rel.as_posix()} imports FastAPI/Starlette outside core/security.py"
                )

    assert not violations, "Backend layer violations found:\n" + "\n".join(violations)
