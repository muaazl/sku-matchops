"""
SKU MatchOps - Unified Model, Catalog & Classifier Sync CLI

Single management script covering every offline/online setup & rebuild task:

  --models       {sync,export,quantize,rebuild}
                 Exports HF EMBEDDING_MODEL / RERANKER_MODEL / GLiNER to ONNX & INT8.
                 Runs FP32 vs INT8 parity verification automatically on quantization,
                 refreshes EMBEDDING_DIM, and clears stale embedding caches on change.
  --model-target {all,embedding,reranker,gliner}  (default: all)
  --db           {sync,rebuild}  PostgreSQL catalog tables (Sheets/Sample -> Postgres)
  --cache        {sync,rebuild}  Local disk caches (Feather mmap, dicts, NLP metadata)
  --qdrant       {sync,rebuild}  Qdrant hybrid vectors & classifier models
  --arcface      {sync,rebuild}  ArcFace BasicType metric head training & INT8 ONNX export

Shortcuts:
  (no stage flags)  Sync all 5 stages idempotently (--models/db/cache/qdrant/arcface sync)
  --rebuild         Full rebuild of all 5 stages
  --sample          Offline demo rebuild from data/sample/SampleData.xlsx

Examples:
  python -m scripts.sync
  python -m scripts.sync --rebuild
  python -m scripts.sync --models sync
  python -m scripts.sync --models rebuild --model-target embedding --qdrant rebuild --arcface rebuild
  python -m scripts.sync --db sync --cache rebuild --qdrant rebuild --arcface rebuild
  python -m scripts.sync --sample --domain food
"""

import argparse
import gc
import inspect
import json
import logging
import os
import shutil
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("matchops.sync")

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

TMP_DIR = os.path.join(PROJECT_ROOT, "data", "cache", "onnx_tmp")
_STAGING_DIR = os.path.join(PROJECT_ROOT, "data", "staged_sheets")
_DOMAIN_MARKET = "market"
_DOMAIN_FOOD = "food"

_EMBEDDING_REPO_ALIASES = {
    "jinaai/jina-embeddings-v5-text-small": "jinaai/jina-embeddings-v5-text-small-retrieval",
}


# ---------------------------------------------------------------------------
# Shared Helpers & Cache Management
# ---------------------------------------------------------------------------


def _ensure_dirs() -> None:
    from engine import config

    os.makedirs(config.ONNX_DIR, exist_ok=True)
    os.makedirs(config.EMBEDDING_ROOT_DIR, exist_ok=True)
    os.makedirs(config.RERANKER_ROOT_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)


def _read_manifest(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            pass
    return {}


def _write_manifest(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _remove_onnx_artifacts(base_onnx_path: str) -> None:
    for suffix in ("", ".data", "_data"):
        stale = base_onnx_path + suffix
        if os.path.exists(stale):
            os.remove(stale)


def _qdrant_state_path() -> str:
    from engine import config

    return os.path.join(config.CACHE_DIR, "qdrant_embedding_models.json")


def _get_qdrant_embedding_model(domain: str) -> str:
    state = _read_manifest(_qdrant_state_path())
    return str(state.get(domain, "") or "")


def _record_qdrant_embedding_model(domain: str) -> None:
    from engine import config

    path = _qdrant_state_path()
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    state = _read_manifest(path)
    state[domain] = config.EMBEDDING_MODEL
    _write_manifest(path, state)


def _active_emb_marker_path() -> str:
    from engine import config

    return os.path.join(config.CACHE_DIR, "active_embedding_model.txt")


def _record_active_embedding_model() -> None:
    from engine import config

    os.makedirs(config.CACHE_DIR, exist_ok=True)
    with open(_active_emb_marker_path(), "w", encoding="utf-8") as f:
        f.write(config.EMBEDDING_MODEL)


def _ensure_clean_embedding_caches_for_model() -> None:
    """Wipes embedding caches if the active EMBEDDING_MODEL in .env differs from cached vectors."""
    from engine import config

    marker = _active_emb_marker_path()
    if not os.path.exists(marker):
        return
    try:
        previous = open(marker, encoding="utf-8").read().strip()
    except Exception:
        previous = ""
    if previous and previous != config.EMBEDDING_MODEL:
        logger.info(
            f"[CACHE] Embedding model changed ({previous} -> {config.EMBEDDING_MODEL}); "
            f"wiping stale embedding caches."
        )
        wipe_embedding_caches()


def wipe_embedding_caches() -> None:
    """Deletes stale embedding-dependent disk caches before re-syncing an embedding model."""
    from engine import config

    if not os.path.exists(config.CACHE_DIR):
        return
    removed = 0
    stale_suffixes = (
        "_weighted_skus_cache.pkl",
        "_incremental.pkl",
        "_warm_centroids.pkl",
        "_classifier_model.pkl",
        "_row_hashes.json",
    )
    stale_substrings = ("_classifier_bt_", "_classifier_third_tag_")
    for fname in os.listdir(config.CACHE_DIR):
        fpath = os.path.join(config.CACHE_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        if (
            fname in ("model_hashes.json", "active_embedding_model.txt")
            or fname.endswith(stale_suffixes)
            or any(sub in fname for sub in stale_substrings)
        ):
            try:
                os.remove(fpath)
                removed += 1
            except Exception as e:
                logger.warning(f"Could not remove stale embedding cache {fpath}: {e}")
    logger.info(f"Cleared {removed} stale embedding cache files from {config.CACHE_DIR}.")


# ---------------------------------------------------------------------------
# 1. ONNX Export & INT8 Quantization (--models)
# ---------------------------------------------------------------------------


def _save_consolidated_onnx(tmp_onnx_path: str, final_onnx_path: str) -> None:
    """Moves exported ONNX (+ external data) or consolidates scattered files without ByteSize()."""
    tmp_dir = os.path.dirname(tmp_onnx_path)
    os.makedirs(os.path.dirname(final_onnx_path), exist_ok=True)
    _remove_onnx_artifacts(final_onnx_path)

    tmp_files = {f for f in os.listdir(tmp_dir) if os.path.isfile(os.path.join(tmp_dir, f))}
    base_name = os.path.basename(tmp_onnx_path)
    allowed_single = {base_name, base_name + ".data", base_name + "_data"}

    if tmp_files <= allowed_single:
        shutil.move(tmp_onnx_path, final_onnx_path)
        for suffix in (".data", "_data"):
            src_data = tmp_onnx_path + suffix
            if os.path.exists(src_data):
                shutil.move(src_data, final_onnx_path + suffix)
        return

    import onnx

    total_bytes = sum(os.path.getsize(os.path.join(tmp_dir, f)) for f in tmp_files)
    save_external = total_bytes > int(1.5 * 1024 * 1024 * 1024)
    proto = onnx.load(tmp_onnx_path, load_external_data=True)
    data_path = final_onnx_path + ".data"
    onnx.save_model(
        proto,
        final_onnx_path,
        save_as_external_data=save_external,
        all_tensors_to_one_file=True,
        location=os.path.basename(data_path),
    )


def _torch_onnx_export(
    wrapper,
    sample_args,
    tmp_onnx_path,
    input_names,
    output_names,
    dynamic_axes,
    prefer_dynamo: bool = True,
) -> None:
    """Exports wrapper to tmp_onnx_path, preferring dynamo=True with external_data=True."""
    if prefer_dynamo:
        try:
            torch.onnx.export(
                wrapper,
                sample_args,
                tmp_onnx_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=18,
                dynamo=True,
                external_data=True,
            )
            return
        except Exception as dynamo_err:
            logger.warning(f"Dynamo ONNX export failed ({dynamo_err}); retrying legacy exporter...")

    torch.onnx.export(
        wrapper,
        sample_args,
        tmp_onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
    )


def _resolve_hf_source(repo_id: str) -> str:
    """Resolves a HuggingFace repo ID to a local cached snapshot directory when available."""
    if os.path.isdir(repo_id):
        return repo_id
    try:
        from huggingface_hub import snapshot_download

        local_snap = snapshot_download(repo_id=repo_id, local_files_only=True)
        if os.path.isdir(local_snap):
            return local_snap
    except Exception:
        pass
    return repo_id


def _accepts_kwarg(module, kwarg: str) -> bool:
    try:
        return kwarg in inspect.signature(module.forward).parameters
    except Exception:
        return False


if nn is not None:

    class EmbeddingONNXWrapper(nn.Module):
        """Model-agnostic ONNX wrapper supporting cls/mean/last_token pooling and optional sparse head."""

        def __init__(
            self,
            encoder: nn.Module,
            pooling: str = "mean",
            sparse_linear: nn.Module | None = None,
        ):
            super().__init__()
            self.encoder = encoder
            self.pooling = pooling
            self.sparse_linear = sparse_linear
            self._pass_use_cache = _accepts_kwarg(encoder, "use_cache")
            if hasattr(self.encoder, "config") and hasattr(self.encoder.config, "use_cache"):
                self.encoder.config.use_cache = False

        def forward(self, input_ids, attention_mask):
            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "return_dict": True,
            }
            if self._pass_use_cache:
                kwargs["use_cache"] = False
            outputs = self.encoder(**kwargs)
            hidden_states = (
                outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
            )

            if self.pooling == "cls":
                dense_vecs = hidden_states[:, 0, :]
            elif self.pooling == "last_token":
                positions = torch.arange(
                    attention_mask.size(1), device=attention_mask.device
                ).unsqueeze(0)
                last_idx = torch.clamp((attention_mask * (positions + 1)).argmax(dim=1), min=0)
                batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
                dense_vecs = hidden_states[batch_idx, last_idx]
            else:
                mask = attention_mask.unsqueeze(-1).type_as(hidden_states)
                summed = (hidden_states * mask).sum(dim=1)
                counts = torch.clamp(mask.sum(dim=1), min=1e-9)
                dense_vecs = summed / counts

            if self.sparse_linear is not None:
                sparse_weights = torch.relu(self.sparse_linear(hidden_states)).squeeze(-1)
                return dense_vecs, sparse_weights
            return dense_vecs

    class RerankerONNXWrapper(nn.Module):
        """Model-agnostic ONNX wrapper supporting listwise LBNL rerankers and pairwise cross-encoders."""

        def __init__(self, model: nn.Module, arch_type: str = "cross_encoder"):
            super().__init__()
            self.model = model
            self.arch_type = arch_type
            if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = False

            self.backbone = getattr(model, "model", None) if arch_type == "listwise_lbnl" else None
            self.projector = (
                getattr(model, "projector", None) if arch_type == "listwise_lbnl" else None
            )
            if (
                self.backbone is not None
                and hasattr(self.backbone, "config")
                and hasattr(self.backbone.config, "use_cache")
            ):
                self.backbone.config.use_cache = False

            target_mod = self.backbone if self.backbone is not None else self.model
            self._pass_use_cache = _accepts_kwarg(target_mod, "use_cache")

        def forward(self, input_ids, attention_mask):
            if self.arch_type == "listwise_lbnl":
                if self.backbone is not None and self.projector is not None:
                    kwargs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "return_dict": True,
                    }
                    if self._pass_use_cache:
                        kwargs["use_cache"] = False
                    outputs = self.backbone(**kwargs)
                    hidden_states = (
                        outputs.last_hidden_state
                        if hasattr(outputs, "last_hidden_state")
                        else outputs[0]
                    )
                    return self.projector(hidden_states)

                kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "output_hidden_states": True,
                    "return_dict": True,
                }
                if self._pass_use_cache:
                    kwargs["use_cache"] = False
                outputs = self.model(**kwargs)
                hidden_states = (
                    outputs.hidden_states[-1]
                    if getattr(outputs, "hidden_states", None)
                    else outputs.last_hidden_state
                )
                return (
                    self.projector(hidden_states) if self.projector is not None else hidden_states
                )

            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )
            return outputs.logits if hasattr(outputs, "logits") else outputs[0]


def _resolve_hf_embedding_id(model_id: str) -> str:
    return _EMBEDDING_REPO_ALIASES.get(model_id, model_id)


def _detect_pooling_mode(source_path: str, hf_id: str, model_id: str, hf_config) -> str:
    from engine import config

    override = getattr(config, "EMBEDDING_POOLING", "auto")
    if override in ("cls", "mean", "last_token"):
        return override

    pooling_cfg = None
    if os.path.isdir(source_path):
        cand = os.path.join(source_path, "1_Pooling", "config.json")
        if os.path.exists(cand):
            pooling_cfg = cand
    else:
        try:
            from huggingface_hub import hf_hub_download

            pooling_cfg = hf_hub_download(repo_id=hf_id, filename="1_Pooling/config.json")
        except Exception:
            pooling_cfg = None

    if pooling_cfg and os.path.exists(pooling_cfg):
        try:
            with open(pooling_cfg, encoding="utf-8") as f:
                pc = json.load(f)
            if pc.get("pooling_mode_lasttoken_tokens"):
                return "last_token"
            if pc.get("pooling_mode_cls_token"):
                return "cls"
            if pc.get("pooling_mode_mean_tokens") or pc.get("pooling_mode_mean_sqrt_len_tokens"):
                return "mean"
        except Exception:
            pass

    model_type = str(getattr(hf_config, "model_type", "") or "").lower()
    if model_type in {"qwen2", "qwen3", "llama", "mistral", "gemma", "gemma2"}:
        return "last_token"
    if "bge" in model_id.lower():
        return "cls"
    return "mean"


def _load_sparse_linear(source_path: str, hf_id: str, model_id: str, hidden_size: int):
    if "bge-m3" not in model_id.lower():
        return None

    sparse_pt = None
    if os.path.isdir(source_path):
        cand = os.path.join(source_path, "sparse_linear.pt")
        if os.path.exists(cand):
            sparse_pt = cand
    if sparse_pt is None:
        try:
            from huggingface_hub import hf_hub_download

            sparse_pt = hf_hub_download(repo_id=hf_id, filename="sparse_linear.pt")
        except Exception:
            sparse_pt = None

    if not sparse_pt or not os.path.exists(sparse_pt):
        raise FileNotFoundError(
            f"Expected 'sparse_linear.pt' for BGE-M3 model '{model_id}', but could not locate it."
        )

    linear = nn.Linear(hidden_size, 1)
    state = torch.load(sparse_pt, map_location="cpu", weights_only=True)
    linear.load_state_dict(state)
    linear.eval()
    return linear


def _detect_reranker_arch(model_id: str, hf_config) -> str:
    archs = [str(a) for a in (getattr(hf_config, "architectures", None) or [])]
    if "JinaForRanking" in archs or "jina-reranker-v3" in model_id.lower():
        return "listwise_lbnl"
    return "cross_encoder"


def _has_saved_tokenizer(target_dir: str) -> bool:
    return os.path.exists(os.path.join(target_dir, "tokenizer.json")) or os.path.exists(
        os.path.join(target_dir, "tokenizer_config.json")
    )


def export_embedding(force: bool = False, require_int8: bool = True) -> bool:
    """Exports configured EMBEDDING_MODEL to FP32 ONNX. Returns True if a new export ran."""
    from engine import config

    target_dir = config.EMBEDDING_DIR
    onnx_fp32 = config.EMBEDDING_ONNX_FP32
    onnx_int8 = config.EMBEDDING_ONNX_INT8
    manifest = _read_manifest(config.EMBEDDING_MANIFEST)
    artifact_ready = os.path.exists(onnx_int8) if require_int8 else os.path.exists(onnx_fp32)
    explicit_pooling = getattr(config, "EMBEDDING_POOLING", "auto")
    pooling_matches = explicit_pooling not in ("cls", "mean", "last_token") or (
        manifest.get("pooling") == explicit_pooling
    )

    if (
        not force
        and manifest.get("model_name") == config.EMBEDDING_MODEL
        and pooling_matches
        and artifact_ready
        and _has_saved_tokenizer(target_dir)
    ):
        logger.info(f"Embedding ({config.EMBEDDING_MODEL}) ONNX already exists. Skipping export.")
        return False

    from transformers import AutoConfig, AutoModel, AutoTokenizer

    model_id = config.EMBEDDING_MODEL
    hf_id = _resolve_hf_embedding_id(model_id)
    source = _resolve_hf_source(hf_id)
    local_only = os.path.isdir(source)
    logger.info(f"Exporting Embedding model '{model_id}' (source: '{hf_id}') to ONNX...")

    os.makedirs(target_dir, exist_ok=True)
    _remove_onnx_artifacts(onnx_int8)
    _remove_onnx_artifacts(onnx_fp32)

    tmp_sub = os.path.join(TMP_DIR, "embedding")
    shutil.rmtree(tmp_sub, ignore_errors=True)
    os.makedirs(tmp_sub, exist_ok=True)
    tmp_onnx = os.path.join(tmp_sub, "model.onnx")

    hf_config = AutoConfig.from_pretrained(
        source, trust_remote_code=True, local_files_only=local_only
    )
    pooling = _detect_pooling_mode(source, hf_id, model_id, hf_config)
    hidden_size = int(getattr(hf_config, "hidden_size", 1024))
    sparse_linear = _load_sparse_linear(source, hf_id, model_id, hidden_size)
    has_sparse = sparse_linear is not None

    tokenizer = AutoTokenizer.from_pretrained(
        source, trust_remote_code=True, local_files_only=local_only, fix_mistral_regex=True
    )
    tokenizer.padding_side = "right"
    hf_model = AutoModel.from_pretrained(
        source, trust_remote_code=True, local_files_only=local_only
    )
    hf_model.eval()
    wrapper = EmbeddingONNXWrapper(hf_model, pooling=pooling, sparse_linear=sparse_linear)
    wrapper.eval()

    inputs = tokenizer(["Test SKU item"], padding=True, truncation=True, return_tensors="pt")
    output_names = ["dense_vecs", "sparse_weights"] if has_sparse else ["dense_vecs"]
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "dense_vecs": {0: "batch_size"},
    }
    if has_sparse:
        dynamic_axes["sparse_weights"] = {0: "batch_size", 1: "sequence_length"}

    _torch_onnx_export(
        wrapper,
        (inputs["input_ids"], inputs["attention_mask"]),
        tmp_onnx,
        input_names=["input_ids", "attention_mask"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        prefer_dynamo=not has_sparse,
    )

    # Free PyTorch model memory before onnx.load/save consolidation
    del hf_model, wrapper, sparse_linear, inputs
    gc.collect()

    _save_consolidated_onnx(tmp_onnx, onnx_fp32)
    tokenizer.save_pretrained(target_dir)
    hf_config.save_pretrained(target_dir)
    _write_manifest(
        config.EMBEDDING_MANIFEST,
        {
            "model_name": model_id,
            "hf_repo": hf_id,
            "pooling": pooling,
            "embedding_dim": hidden_size,
            "has_learned_sparse": has_sparse,
        },
    )
    del tokenizer
    gc.collect()
    logger.info(
        f"Embedding exported to {onnx_fp32} (dim={hidden_size}, pooling={pooling}, learned_sparse={has_sparse})"
    )
    return True


def export_reranker(force: bool = False, require_int8: bool = True) -> bool:
    """Exports configured RERANKER_MODEL to FP32 ONNX. Returns True if a new export ran."""
    from engine import config

    target_dir = config.RERANKER_DIR
    onnx_fp32 = config.RERANKER_ONNX_FP32
    onnx_int8 = config.RERANKER_ONNX_INT8
    manifest = _read_manifest(config.RERANKER_MANIFEST)
    artifact_ready = os.path.exists(onnx_int8) if require_int8 else os.path.exists(onnx_fp32)

    if (
        not force
        and manifest.get("model_name") == config.RERANKER_MODEL
        and artifact_ready
        and _has_saved_tokenizer(target_dir)
    ):
        logger.info(f"Reranker ({config.RERANKER_MODEL}) ONNX already exists. Skipping export.")
        return False

    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    model_id = config.RERANKER_MODEL
    source = _resolve_hf_source(model_id)
    local_only = os.path.isdir(source)
    logger.info(f"Exporting Reranker '{model_id}' to ONNX...")

    os.makedirs(target_dir, exist_ok=True)
    _remove_onnx_artifacts(onnx_int8)
    _remove_onnx_artifacts(onnx_fp32)

    tmp_sub = os.path.join(TMP_DIR, "reranker")
    shutil.rmtree(tmp_sub, ignore_errors=True)
    os.makedirs(tmp_sub, exist_ok=True)
    tmp_onnx = os.path.join(tmp_sub, "model.onnx")

    hf_config = AutoConfig.from_pretrained(
        source, trust_remote_code=True, local_files_only=local_only
    )
    arch_type = _detect_reranker_arch(model_id, hf_config)
    tokenizer = AutoTokenizer.from_pretrained(
        source, trust_remote_code=True, local_files_only=local_only, fix_mistral_regex=True
    )

    if arch_type == "listwise_lbnl":
        hf_model = AutoModel.from_pretrained(
            source, trust_remote_code=True, local_files_only=local_only
        )
        hf_model.eval()
        wrapper = RerankerONNXWrapper(hf_model, arch_type="listwise_lbnl")
        wrapper.eval()

        spec_tokens = getattr(hf_model, "special_tokens", {}) or {}
        doc_tok = spec_tokens.get("doc_embed_token", "<|embed_token|>")
        query_tok = spec_tokens.get("query_embed_token", "<|rerank_token|>")
        sample_prompt = (
            f'<passage id="0">\nDoc{doc_tok}\n</passage>\n<query>\nQuery{query_tok}\n</query>'
        )
        inputs = tokenizer([sample_prompt], padding=True, truncation=True, return_tensors="pt")
        _torch_onnx_export(
            wrapper,
            (inputs["input_ids"], inputs["attention_mask"]),
            tmp_onnx,
            input_names=["input_ids", "attention_mask"],
            output_names=["projected_embeds"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "projected_embeds": {0: "batch_size", 1: "sequence_length"},
            },
        )
        unk_id = getattr(tokenizer, "unk_token_id", None)
        doc_id = tokenizer.convert_tokens_to_ids(doc_tok)
        query_id = tokenizer.convert_tokens_to_ids(query_tok)
        manifest_payload = {
            "model_name": model_id,
            "arch_type": "listwise_lbnl",
            "score_activation": "identity",
            "doc_embed_token": doc_tok,
            "query_embed_token": query_tok,
            "doc_embed_token_id": (
                int(doc_id)
                if isinstance(doc_id, int) and doc_id != unk_id
                else int(getattr(hf_model, "doc_embed_token_id", 151670))
            ),
            "query_embed_token_id": (
                int(query_id)
                if isinstance(query_id, int) and query_id != unk_id
                else int(getattr(hf_model, "query_embed_token_id", 151671))
            ),
        }
    else:
        hf_model = AutoModelForSequenceClassification.from_pretrained(
            source, trust_remote_code=True, local_files_only=local_only
        )
        hf_model.eval()
        wrapper = RerankerONNXWrapper(hf_model, arch_type="cross_encoder")
        wrapper.eval()
        inputs = tokenizer(
            ["query"], ["passage"], padding=True, truncation=True, return_tensors="pt"
        )
        _torch_onnx_export(
            wrapper,
            (inputs["input_ids"], inputs["attention_mask"]),
            tmp_onnx,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size"},
            },
        )
        manifest_payload = {
            "model_name": model_id,
            "arch_type": "cross_encoder",
            "score_activation": "sigmoid",
        }

    del hf_model, wrapper, inputs
    gc.collect()

    _save_consolidated_onnx(tmp_onnx, onnx_fp32)
    tokenizer.save_pretrained(target_dir)
    hf_config.save_pretrained(target_dir)
    _write_manifest(config.RERANKER_MANIFEST, manifest_payload)
    del tokenizer
    gc.collect()
    logger.info(f"Reranker exported to {onnx_fp32} (arch={arch_type})")
    return True


def export_gliner(force: bool = False) -> None:
    from engine import config

    target_dir = config.GLINER_MODEL
    onnx_file = config.GLINER_ONNX
    config_file = os.path.join(target_dir, "gliner_config.json")

    if not force and os.path.exists(onnx_file) and os.path.exists(config_file):
        logger.info("GLiNER ONNX model already exists. Skipping download.")
        return

    from huggingface_hub import snapshot_download

    logger.info("Downloading GLiNER Medium v2.1 ONNX model...")
    os.makedirs(target_dir, exist_ok=True)
    snapshot_download(
        repo_id="onnx-community/gliner_medium-v2.1",
        local_dir=target_dir,
        allow_patterns=["*.json", "*.txt", "model.onnx", "onnx/model.onnx"],
    )

    nested_onnx = os.path.join(target_dir, "onnx", "model.onnx")
    if os.path.exists(nested_onnx):
        shutil.move(nested_onnx, onnx_file)
        shutil.rmtree(os.path.join(target_dir, "onnx"), ignore_errors=True)

    gc.collect()
    logger.info(f"GLiNER ONNX model ready at {onnx_file}")


def _filter_ort_inputs(session, ort_inputs: dict) -> dict:
    valid_names = {inp.name for inp in session.get_inputs()}
    return {k: v for k, v in ort_inputs.items() if k in valid_names}


def quantize_model(input_path: str, output_path: str, label: str) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Source FP32 model not found at: {input_path}")

    logger.info(f"[{label}] Starting INT8 dynamic quantization from {input_path}...")
    start_t = time.time()
    try:
        quantize_dynamic(
            model_input=input_path,
            model_output=output_path,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
            extra_options={"MatMulConstBOnly": True},
        )
    except Exception as quant_err:
        err_str = str(quant_err).lower()
        if (
            "encodeerror" in type(quant_err).__name__.lower()
            or "serialize" in err_str
            or "2gb" in err_str
        ):
            logger.info(
                f"[{label}] Single-file protobuf limit reached ({quant_err}); retrying with external data..."
            )
            quantize_dynamic(
                model_input=input_path,
                model_output=output_path,
                weight_type=QuantType.QInt8,
                op_types_to_quantize=["MatMul", "Gemm"],
                use_external_data_format=True,
                extra_options={"MatMulConstBOnly": True},
            )
        else:
            raise

    def _size_mb(path: str) -> float:
        total = os.path.getsize(path)
        for suffix in ("_data", ".data"):
            if os.path.exists(path + suffix):
                total += os.path.getsize(path + suffix)
        return total / (1024 * 1024)

    in_mb = _size_mb(input_path)
    out_mb = _size_mb(output_path)
    reduction = (1.0 - (out_mb / in_mb)) * 100.0 if in_mb > 0 else 0.0
    logger.info(
        f"[{label}] Quantized in {time.time() - start_t:.2f}s: "
        f"{in_mb:.1f} MB -> {out_mb:.1f} MB ({reduction:.1f}% reduction)"
    )


def _verify_embedding_parity() -> None:
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    from engine import config

    fp32_model = config.EMBEDDING_ONNX_FP32
    int8_model = config.EMBEDDING_ONNX_INT8
    if not (os.path.exists(fp32_model) and os.path.exists(int8_model)):
        return

    manifest = config.get_embedding_manifest()
    pooling = manifest.get("pooling", "mean")
    logger.info(
        f"[VERIFY] Checking Embedding ({config.EMBEDDING_MODEL}, pooling={pooling}) INT8 vs FP32..."
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.EMBEDDING_DIR,
        local_files_only=True,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    tokenizer.padding_side = "right"
    sample_texts = [
        "Red Bull Energy Drink 250ml",
        "Coca Cola Zero Sugar 330ml Can",
        "Fresh organic bananas 1kg",
    ]
    inputs = tokenizer(sample_texts, padding=True, truncation=True, return_tensors="np")

    t0 = time.time()
    fp32_sess = ort.InferenceSession(fp32_model, providers=["CPUExecutionProvider"])
    fp32_load = time.time() - t0
    ort_inputs = _filter_ort_inputs(fp32_sess, {k: v.astype(np.int64) for k, v in inputs.items()})
    t0 = time.time()
    fp32_out = fp32_sess.run(None, ort_inputs)[0]
    fp32_inf = (time.time() - t0) * 1000

    t0 = time.time()
    int8_sess = ort.InferenceSession(int8_model, providers=["CPUExecutionProvider"])
    int8_load = time.time() - t0
    t0 = time.time()
    int8_out = int8_sess.run(None, ort_inputs)[0]
    int8_inf = (time.time() - t0) * 1000

    if fp32_out.ndim == 3:
        from engine.nlp.embedding_engine import EmbeddingEngine

        attn = inputs["attention_mask"]
        fp32_out = EmbeddingEngine._pool_hidden_states(fp32_out, attn, pooling)
        int8_out = EmbeddingEngine._pool_hidden_states(int8_out, attn, pooling)

    del fp32_sess, int8_sess
    gc.collect()

    if not np.isfinite(int8_out).all():
        raise RuntimeError(
            "[VERIFY] Embedding INT8 produced NaN/Inf outputs; aborting FP32 cleanup."
        )

    fp32_norms = np.linalg.norm(fp32_out, axis=1, keepdims=True)
    int8_norms = np.linalg.norm(int8_out, axis=1, keepdims=True)
    if (fp32_norms == 0).any() or (int8_norms == 0).any():
        raise RuntimeError("[VERIFY] Embedding produced zero-norm vectors; aborting FP32 cleanup.")

    fp32_norm = fp32_out / fp32_norms
    int8_norm = int8_out / int8_norms
    cos_sims = np.sum(fp32_norm * int8_norm, axis=1)
    min_cos = float(np.min(cos_sims))
    logger.info(
        f"[VERIFY] Embedding Load: FP32={fp32_load:.2f}s INT8={int8_load:.2f}s | "
        f"Infer: FP32={fp32_inf:.1f}ms INT8={int8_inf:.1f}ms | "
        f"Cosine={[round(float(s), 4) for s in cos_sims]}"
    )
    if min_cos < 0.90:
        raise RuntimeError(
            f"[VERIFY] Embedding INT8 cosine parity failed (min_cos={min_cos:.4f} < 0.90)."
        )


def _verify_reranker_parity() -> None:
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    from engine import config

    fp32_model = config.RERANKER_ONNX_FP32
    int8_model = config.RERANKER_ONNX_INT8
    if not (os.path.exists(fp32_model) and os.path.exists(int8_model)):
        return

    manifest = config.get_reranker_manifest()
    arch = manifest.get("arch_type", "cross_encoder")
    logger.info(
        f"[VERIFY] Checking Reranker ({config.RERANKER_MODEL}, arch={arch}) INT8 vs FP32..."
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.RERANKER_DIR,
        local_files_only=True,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )

    t0 = time.time()
    fp32_sess = ort.InferenceSession(fp32_model, providers=["CPUExecutionProvider"])
    fp32_load = time.time() - t0
    t0 = time.time()
    int8_sess = ort.InferenceSession(int8_model, providers=["CPUExecutionProvider"])
    int8_load = time.time() - t0

    try:
        if arch == "listwise_lbnl":
            doc_tok = manifest.get("doc_embed_token", "<|embed_token|>")
            query_tok = manifest.get("query_embed_token", "<|rerank_token|>")
            doc_id = int(manifest.get("doc_embed_token_id", 151670))
            query_id = int(manifest.get("query_embed_token_id", 151671))
            prompt = (
                f'<passage id="0">\nName: Red Bull Energy Drink 250ml{doc_tok}\n</passage>\n'
                f'<passage id="1">\nName: Organic Bananas 1kg{doc_tok}\n</passage>\n'
                f"<query>\nName: Red Bull 250ml{query_tok}\n</query>"
            )
            inputs = tokenizer([prompt], return_tensors="np")
            ort_inputs = _filter_ort_inputs(
                fp32_sess, {k: v.astype(np.int64) for k, v in inputs.items()}
            )
            fp32_out = fp32_sess.run(None, ort_inputs)[0][0]
            int8_out = int8_sess.run(None, ort_inputs)[0][0]
            if not np.isfinite(int8_out).all():
                raise RuntimeError(
                    "[VERIFY] Reranker INT8 produced NaN/Inf; aborting FP32 cleanup."
                )
            doc_pos = np.where(inputs["input_ids"][0] == doc_id)[0]
            q_pos_arr = np.where(inputs["input_ids"][0] == query_id)[0]
            if len(doc_pos) < 2 or len(q_pos_arr) == 0:
                raise RuntimeError(
                    f"[VERIFY] Reranker missing special token IDs in tokenized prompt "
                    f"(doc_count={len(doc_pos)}, query_count={len(q_pos_arr)})."
                )
            q_pos = int(q_pos_arr[-1])

            def _cos(emb_seq):
                q_vec = emb_seq[q_pos]
                d_mat = emb_seq[doc_pos]
                q_norm = np.linalg.norm(q_vec)
                d_norms = np.linalg.norm(d_mat, axis=1, keepdims=True)
                if q_norm == 0 or (d_norms == 0).any():
                    raise RuntimeError("[VERIFY] Reranker produced zero-norm embeddings.")
                return (d_mat / d_norms) @ (q_vec / q_norm)

            fp32_scores = _cos(fp32_out)
            int8_scores = _cos(int8_out)
            logger.info(
                f"[VERIFY] Reranker Load: FP32={fp32_load:.2f}s INT8={int8_load:.2f}s | "
                f"FP32={[round(float(v), 4) for v in fp32_scores]} "
                f"INT8={[round(float(v), 4) for v in int8_scores]}"
            )
            if (
                int8_scores[0] <= int8_scores[1]
                or float(np.max(np.abs(fp32_scores - int8_scores))) > 0.25
            ):
                raise RuntimeError(
                    f"[VERIFY] Listwise reranker INT8 parity failed (fp32={fp32_scores.tolist()}, int8={int8_scores.tolist()})."
                )
        else:
            pairs = [
                ("Red Bull Energy Drink", "Red Bull Energy Drink 250ml Can"),
                ("Coca Cola Zero", "Organic Bananas 1kg"),
            ]
            inputs = tokenizer(
                [p[0] for p in pairs],
                [p[1] for p in pairs],
                padding=True,
                truncation=True,
                return_tensors="np",
            )
            ort_inputs = _filter_ort_inputs(
                fp32_sess, {k: v.astype(np.int64) for k, v in inputs.items()}
            )

            def _extract_scores(raw):
                arr = np.asarray(raw)
                return arr[:, 1] if (arr.ndim == 2 and arr.shape[1] == 2) else arr.reshape(-1)

            fp32_logits = _extract_scores(fp32_sess.run(None, ort_inputs)[0])
            int8_logits = _extract_scores(int8_sess.run(None, ort_inputs)[0])
            if not np.isfinite(int8_logits).all():
                raise RuntimeError(
                    "[VERIFY] Reranker INT8 produced NaN/Inf; aborting FP32 cleanup."
                )
            logger.info(
                f"[VERIFY] Reranker Load: FP32={fp32_load:.2f}s INT8={int8_load:.2f}s | "
                f"FP32={[round(float(v), 4) for v in fp32_logits]} "
                f"INT8={[round(float(v), 4) for v in int8_logits]}"
            )
            if int8_logits[0] <= int8_logits[1]:
                raise RuntimeError(
                    f"[VERIFY] Cross-encoder INT8 ranking parity failed (int8={int8_logits.tolist()})."
                )
    finally:
        del fp32_sess, int8_sess
        gc.collect()


def quantize_models(target: str = "all", force: bool = False, cleanup_fp32: bool = True) -> None:
    """Quantizes FP32 ONNX models to INT8, runs parity verification, and removes intermediate FP32."""
    from engine import config

    if target == "gliner":
        logger.info(
            "[GLINER] Staying on FP32 ONNX (model.onnx); INT8 skipped to preserve NER accuracy."
        )
        return

    jobs = []
    if target in ("all", "embedding"):
        jobs.append(
            (
                config.EMBEDDING_ONNX_FP32,
                config.EMBEDDING_ONNX_INT8,
                "EMBEDDING",
                _verify_embedding_parity,
            )
        )
    if target in ("all", "reranker"):
        jobs.append(
            (
                config.RERANKER_ONNX_FP32,
                config.RERANKER_ONNX_INT8,
                "RERANKER",
                _verify_reranker_parity,
            )
        )

    for in_path, out_path, name, verify_fn in jobs:
        ran_quant = False
        if force or not os.path.exists(out_path):
            if not os.path.exists(in_path):
                raise FileNotFoundError(
                    f"Cannot quantize {name}: intermediate FP32 missing at {in_path}."
                )
            quantize_model(in_path, out_path, name)
            ran_quant = True
        else:
            logger.info(f"[{name}] INT8 ONNX already exists ({out_path}). Skipping quantization.")

        if ran_quant or os.path.exists(in_path):
            try:
                verify_fn()
            except Exception:
                if os.path.exists(out_path):
                    _remove_onnx_artifacts(out_path)
                    logger.warning(f"[{name}] Removed failed INT8 artifact: {out_path}")
                raise

        if cleanup_fp32 and os.path.exists(out_path) and os.path.exists(in_path):
            gc.collect()
            _remove_onnx_artifacts(in_path)
            logger.info(f"[{name}] Cleaned intermediate FP32 artifact: {in_path}")


def sync_models(
    mode: str,
    target: str = "all",
    _is_quantize_sub: bool = False,
) -> None:
    """Orchestrates ONNX export + isolated-subprocess INT8 quantization + verification."""
    from engine import config

    if _is_quantize_sub or mode == "quantize":
        quantize_models(target=target, force=(mode != "quantize"), cleanup_fp32=True)
        config.refresh_model_config()
        return

    _ensure_dirs()
    force = mode == "rebuild"
    require_int8 = mode != "export"

    exported_emb = False
    exported_rerank = False
    try:
        if target in ("all", "embedding"):
            exported_emb = export_embedding(force=force, require_int8=require_int8)
        if target in ("all", "reranker"):
            exported_rerank = export_reranker(force=force, require_int8=require_int8)
        if target in ("all", "gliner"):
            export_gliner(force=force)
    finally:
        if os.path.exists(TMP_DIR):
            shutil.rmtree(TMP_DIR, ignore_errors=True)

    if mode == "export":
        config.refresh_model_config()
        return

    need_quant = (
        exported_emb
        or exported_rerank
        or (target in ("all", "embedding") and not os.path.exists(config.EMBEDDING_ONNX_INT8))
        or (target in ("all", "reranker") and not os.path.exists(config.RERANKER_ONNX_INT8))
    )
    if need_quant:
        logger.info("Running INT8 quantization + parity verification in an isolated subprocess...")
        env = os.environ.copy()
        env["PYTHONPATH"] = (
            f"{PROJECT_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
            if "PYTHONPATH" in env
            else PROJECT_ROOT
        )
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--models",
            "quantize",
            "--model-target",
            target,
            "--_quantize-sub",
        ]
        subprocess.run(cmd, env=env, check=True)

    config.refresh_model_config()

    if target in ("all", "embedding") and (force or exported_emb):
        wipe_embedding_caches()


# ---------------------------------------------------------------------------
# 2. PostgreSQL (--db), Disk Cache (--cache), Qdrant/Classifiers (--qdrant)
# ---------------------------------------------------------------------------


def _resolve_project_path(path: str) -> str:
    if os.path.isabs(path) or os.path.exists(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def _get_staged_path(sheet_name: str) -> str:
    safe_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in sheet_name) + ".csv"
    return os.path.join(_STAGING_DIR, safe_name)


def _domain_sheet_names(domain: str) -> list[str]:
    if domain == _DOMAIN_FOOD:
        return ["Food Catalog", "Food_Flavors", "Food_GK", "Food_BT", "Food_Region"]
    return ["Market Catalog", "Market_Brands", "Market_GK", "Market_BT", "Market_Category"]


def _write_bt_gk_map_csv(cat_df, domain: str) -> None:
    """Derives Classifier_BT_GK_Map_{Market,Food}.csv from a catalog DataFrame."""
    col_bt = next(
        (c for c in cat_df.columns if str(c).strip().lower() in ("basictype", "basic type")),
        None,
    )
    col_gk = next(
        (
            c
            for c in cat_df.columns
            if str(c).strip().lower() in ("generickeywords", "generic keywords")
        ),
        None,
    )
    if not (col_bt and col_gk):
        return
    suffix = "Food" if domain == _DOMAIN_FOOD else "Market"
    grouped = (
        cat_df[[col_bt, col_gk]]
        .dropna()
        .drop_duplicates()
        .groupby(col_bt)[col_gk]
        .apply(lambda x: ", ".join([k.strip() for k in x.unique() if str(k).strip()]))
        .reset_index()
    )
    grouped.columns = ["basictype", "generic keywords"]
    grouped.to_csv(
        _get_staged_path(f"Classifier_BT_GK_Map_{suffix}"),
        index=False,
        encoding="utf-8",
    )


def _stage_all_sheets(sheet_id: str, domains: list[str]) -> None:
    """Downloads Google Sheet tabs as CSVs into _STAGING_DIR without importing engine."""
    import pandas as pd
    import requests

    os.makedirs(_STAGING_DIR, exist_ok=True)
    tabs: list[str] = []
    for d in domains:
        for tab in _domain_sheet_names(d):
            if tab not in tabs:
                tabs.append(tab)

    for idx, tab in enumerate(tabs, start=1):
        encoded = tab.replace(" ", "%20")
        url = (
            f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={encoded}"
        )
        text = None
        last_err = None
        for attempt in range(1, 4):
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                if not r.text or not r.text.strip():
                    raise ValueError(f"Empty sheet response for '{tab}'")
                text = r.text
                break
            except Exception as e:
                last_err = e
                if attempt < 3:
                    time.sleep(1.5**attempt)
        if text is None:
            raise ConnectionError(f"Failed to download sheet '{tab}': {last_err}") from last_err
        with open(_get_staged_path(tab), "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f"[STAGE] ({idx}/{len(tabs)}) Staged '{tab}'.")

    for d in domains:
        cat_name = "Food Catalog" if d == _DOMAIN_FOOD else "Market Catalog"
        cat_path = _get_staged_path(cat_name)
        if os.path.exists(cat_path):
            _write_bt_gk_map_csv(pd.read_csv(cat_path, dtype=str), d)


def _stage_all_from_excel(excel_path: str, domains: list[str]) -> None:
    """Extracts Excel workbook tabs as CSVs into _STAGING_DIR without importing engine."""
    import pandas as pd

    os.makedirs(_STAGING_DIR, exist_ok=True)
    xl = pd.ExcelFile(excel_path)
    sheet_names = xl.sheet_names
    for name in sheet_names:
        df = xl.parse(name)
        df.to_csv(_get_staged_path(name), index=False, encoding="utf-8")
        if name == "Market_Cat":
            df.to_csv(_get_staged_path("Market_Category"), index=False, encoding="utf-8")
        elif name == "Market_Category":
            df.to_csv(_get_staged_path("Market_Cat"), index=False, encoding="utf-8")

    for d in domains:
        suffix = "Food" if d == _DOMAIN_FOOD else "Market"
        map_name = f"Classifier_BT_GK_Map_{suffix}"
        cat_name = "Food Catalog" if d == _DOMAIN_FOOD else "Market Catalog"
        if map_name not in sheet_names and cat_name in sheet_names:
            _write_bt_gk_map_csv(xl.parse(cat_name), d)


def _cleanup_staged_sheets() -> None:
    if os.path.exists(_STAGING_DIR):
        shutil.rmtree(_STAGING_DIR, ignore_errors=True)
        logger.info(f"[STAGE] Removed temporary staging folder: {_STAGING_DIR}")


def reset_catalog_tables(domain: str | None = None) -> None:
    """Empties catalog tables (for one domain, or all) before a clean import."""
    from sqlalchemy import text

    from backend.app.db.init import init_db
    from backend.app.db.session import get_engine

    init_db()
    try:
        with get_engine().begin() as conn:
            if domain and domain in (_DOMAIN_MARKET, _DOMAIN_FOOD):
                for table in (
                    "catalog_items",
                    "brand_flavors",
                    "classifier_dictionaries",
                    "bt_gk_map",
                ):
                    conn.execute(
                        text(f"DELETE FROM {table} WHERE domain = :domain"), {"domain": domain}
                    )
                logger.info(
                    f"[{domain.upper()}] Deleted existing table rows for domain '{domain}'."
                )
            else:
                conn.execute(
                    text(
                        "TRUNCATE catalog_items, brand_flavors, classifier_dictionaries, bt_gk_map RESTART IDENTITY"
                    )
                )
                logger.info(
                    "Emptied catalog_items, brand_flavors, classifier_dictionaries, and bt_gk_map tables."
                )
    except Exception as e:
        logger.warning(f"Error while resetting catalog tables: {e}")


def wipe_local_cache_files() -> None:
    from engine import config

    logger.info("Purging local disk cache files...")
    if os.path.exists(config.CACHE_DIR):
        for item in os.listdir(config.CACHE_DIR):
            item_path = os.path.join(config.CACHE_DIR, item)
            if os.path.isfile(item_path):
                try:
                    os.remove(item_path)
                except Exception as e:
                    logger.warning(f"Could not remove cache file {item_path}: {e}")


def wipe_domain_cache_files(domain: str) -> None:
    from engine import config

    logger.info(f"[{domain.upper()}] Purging local disk cache files...")
    if os.path.exists(config.CACHE_DIR):
        for item in os.listdir(config.CACHE_DIR):
            fpath = os.path.join(config.CACHE_DIR, item)
            if item.startswith(f"{domain}_") and os.path.isfile(fpath):
                try:
                    os.remove(fpath)
                except Exception as e:
                    logger.warning(f"Could not remove cache file {item}: {e}")


def sync_db(domain: str, sheet_id: str, mode: str):
    """Syncs or rebuilds Google Sheets/Sample data into PostgreSQL."""
    from backend.app.db.init import init_db
    from backend.app.services.catalog_sync import (
        compute_and_store_dictionary_counts,
        sync_bt_gk_map_to_database,
        sync_catalog_to_database,
        sync_classifier_dictionaries_to_database,
    )

    if mode == "rebuild":
        reset_catalog_tables(domain)
    else:
        init_db()

    cat_df, _ = sync_catalog_to_database(sheet_id, domain=domain)
    sync_classifier_dictionaries_to_database(sheet_id, domain=domain)
    sync_bt_gk_map_to_database(sheet_id, domain=domain, cat_df=cat_df)
    compute_and_store_dictionary_counts(domain, cat_df=cat_df)
    return cat_df


def sync_cache(domain: str, sheet_id: str, mode: str) -> None:
    """Syncs or rebuilds local Feather mmap caches, dictionary JSON, and NLP metadata."""
    import joblib
    import pyarrow.feather as feather

    from engine import config
    from engine.data_pipeline.cache_manager import CacheManager
    from engine.data_pipeline.ingestion import DataIngestion

    os.makedirs(config.CACHE_DIR, exist_ok=True)
    if mode == "rebuild":
        DataIngestion.clear_mem_cache(domain)
    try:
        cat_df, brands_df = DataIngestion.load_catalog(sheet_id, domain=domain, force_fetch=False)
    except RuntimeError as exc:
        raise RuntimeError(
            f"[{domain.upper()}] Local/staged catalog unavailable for --cache ({exc}). "
            f"Run with --from-staged (after backend --db --keep-staged) or --sample."
        ) from exc

    catalog_cache_path = os.path.join(config.CACHE_DIR, f"{domain}_catalog_mmap.feather")
    brands_cache_path = os.path.join(config.CACHE_DIR, f"{domain}_brands_mmap.feather")
    dicts_cache_path = os.path.join(config.CACHE_DIR, f"{domain}_classifier_dicts.json")
    metadata_path = os.path.join(config.CACHE_DIR, f"{domain}_catalog_metadata.pkl")

    rebuild = mode == "rebuild"

    if rebuild or not os.path.exists(catalog_cache_path) or not os.path.exists(brands_cache_path):
        cat_df_feather = cat_df.copy()
        if "entities" in cat_df_feather.columns:
            cat_df_feather = cat_df_feather.drop(columns=["entities"])
        if "weight_val" in cat_df_feather.columns:
            cat_df_feather["weight_val"] = cat_df_feather["weight_val"].astype(str)
        cat_df_feather = cat_df_feather.reset_index(drop=True)
        brands_df_feather = brands_df.reset_index(drop=True)

        feather.write_feather(cat_df_feather, catalog_cache_path, compression="lz4")
        feather.write_feather(brands_df_feather, brands_cache_path, compression="lz4")
        logger.info(
            f"[{domain.upper()}] Feather caches written ({len(cat_df_feather)} catalog items, {len(brands_df_feather)} brands)."
        )
    else:
        logger.info(f"[{domain.upper()}] Feather caches already present — skipped.")

    if rebuild or not os.path.exists(dicts_cache_path):
        try:
            dicts = DataIngestion.load_classifier_dictionaries(
                sheet_id, domain=domain, force_fetch=False
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"[{domain.upper()}] Classifier dictionaries unavailable for --cache ({exc}). "
                f"Run with --from-staged (after backend --db --keep-staged) or --sample."
            ) from exc
        with open(dicts_cache_path, "w", encoding="utf-8") as f:
            json.dump(dicts, f, indent=2)
        logger.info(f"[{domain.upper()}] Classifier dictionaries JSON cached.")
    else:
        logger.info(f"[{domain.upper()}] Classifier dictionaries JSON already present — skipped.")

    if rebuild or not os.path.exists(metadata_path):
        meta_df = CacheManager.populate_basic_nlp_features(cat_df.copy())
        joblib.dump(meta_df, metadata_path)
        logger.info(f"[{domain.upper()}] Catalog NLP metadata pickle saved.")
    else:
        logger.info(f"[{domain.upper()}] Catalog metadata pickle already present — skipped.")


def sync_qdrant(domain: str, mode: str) -> None:
    """Syncs or rebuilds Qdrant vector collections and classifier models."""
    from engine import config
    from engine.core.resource_loader import _get_vector_store, get_classifier, get_pipeline

    _ensure_clean_embedding_caches_for_model()
    vs = _get_vector_store()
    do_full_rebuild = mode == "rebuild"
    collections = [vs._get_collection_name(domain), f"{domain}_tags"]

    if not do_full_rebuild:
        prev_qdrant_model = _get_qdrant_embedding_model(domain)
        model_changed = bool(prev_qdrant_model and prev_qdrant_model != config.EMBEDDING_MODEL)
        schema_mismatch = next(
            (
                f"{col}: {msg}"
                for col in collections
                if (msg := vs.get_collection_schema_mismatch(col))
            ),
            None,
        )
        if model_changed or schema_mismatch:
            reason = schema_mismatch or (
                f"embedding model changed ({prev_qdrant_model} -> {config.EMBEDDING_MODEL})"
            )
            logger.info(f"[{domain.upper()}] Qdrant rebuild triggered automatically ({reason}).")
            do_full_rebuild = True
            for stale_file in (
                os.path.join(config.CACHE_DIR, f"{domain}_row_hashes.json"),
                os.path.join(config.CACHE_DIR, f"{domain}_classifier_model.pkl"),
            ):
                if os.path.exists(stale_file):
                    os.remove(stale_file)

    if do_full_rebuild:
        for col in collections:
            try:
                vs.delete_named_collection(col)
                logger.info(
                    f"[{domain.upper()}] Deleted Qdrant collection '{col}' for clean rebuild."
                )
            except Exception as e:
                logger.warning(f"[{domain.upper()}] Could not delete collection '{col}': {e}")
    else:
        try:
            remote_hashes = vs.get_existing_hashes(domain)
        except Exception as e:
            logger.warning(f"[{domain.upper()}] Could not fetch existing hashes from Qdrant: {e}")
            remote_hashes = {}

        if remote_hashes:
            hash_file = os.path.join(config.CACHE_DIR, f"{domain}_row_hashes.json")
            local_hashes = {}
            if os.path.exists(hash_file):
                try:
                    with open(hash_file, encoding="utf-8") as f:
                        local_hashes = json.load(f)
                except Exception:
                    local_hashes = {}
            merged = {**remote_hashes, **local_hashes}
            os.makedirs(config.CACHE_DIR, exist_ok=True)
            with open(hash_file, "w", encoding="utf-8") as f:
                json.dump(merged, f)
            logger.info(
                f"[{domain.upper()}] Reconciled {len(remote_hashes)} row hashes from Qdrant "
                f"({len(merged)} total known rows)."
            )

    clf_cache_path = os.path.join(config.CACHE_DIR, f"{domain}_classifier_model.pkl")
    need_classifier_train = do_full_rebuild or not os.path.exists(clf_cache_path)

    logger.info(f"[{domain.upper()}] Training classifier & syncing catalog vectors to Qdrant...")
    get_classifier(domain, force_reset=need_classifier_train)
    get_pipeline(domain, check_for_updates=True, force_sync=do_full_rebuild)
    _record_qdrant_embedding_model(domain)
    _record_active_embedding_model()
    logger.info(f"[{domain.upper()}] Qdrant sync complete.")


# ---------------------------------------------------------------------------
# 3. ArcFace BasicType Head (--arcface)
# ---------------------------------------------------------------------------


def load_catalog_data(
    domain: str,
    from_sample: bool = False,
    sample_file: str = "data/sample/SampleData.xlsx",
):
    """Loads and standardizes labeled catalog data for ArcFace BT training."""
    import pandas as pd

    from engine import config

    df = pd.DataFrame()
    if from_sample:
        sample_file = _resolve_project_path(sample_file)
        if not os.path.exists(sample_file):
            raise FileNotFoundError(f"Sample workbook not found: {sample_file}")
        sheet_name = (
            config.FOOD_CATALOG_SHEET
            if domain == config.DOMAIN_FOOD
            else config.MARKET_CATALOG_SHEET
        )
        logger.info(
            f"[{domain.upper()}] Loading ArcFace training data from '{sheet_name}' in {sample_file}..."
        )
        df = pd.read_excel(sample_file, sheet_name=sheet_name)
    else:
        from engine.data_pipeline.ingestion import DataIngestion

        cat_df, _ = DataIngestion.load_catalog(config.GOOGLE_SHEET_ID, domain=domain)
        if not cat_df.empty:
            df = cat_df

    if df.empty:
        raise ValueError(f"[{domain.upper()}] No catalog data found for ArcFace training.")

    col_map = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    bt_col = next((col_map[c] for c in ("basictype", "bt") if c in col_map), None)
    if not bt_col:
        raise ValueError(f"[{domain.upper()}] Catalog missing BasicType column: {list(df.columns)}")

    name_col = next((col_map[c] for c in ("name", "skuname", "itemname") if c in col_map), None)
    if not name_col:
        raise ValueError(f"[{domain.upper()}] Catalog missing Name column: {list(df.columns)}")

    desc_col = next((col_map[c] for c in ("description", "desc") if c in col_map), None)
    cat_col = next((col_map[c] for c in ("category", "cat", "region") if c in col_map), None)
    price_col = col_map.get("price")

    df = df.fillna("")
    clean_df = pd.DataFrame()
    clean_df["basictype"] = df[bt_col].astype(str).str.strip()
    clean_df["Name"] = df[name_col].astype(str).str.strip()
    clean_df["Description"] = (
        df[desc_col].astype(str).str.strip() if desc_col else pd.Series([""] * len(df))
    )
    clean_df["Category"] = (
        df[cat_col].astype(str).str.strip() if cat_col else pd.Series([""] * len(df))
    )
    clean_df["Price"] = (
        pd.to_numeric(df[price_col], errors="coerce").fillna(0.0)
        if price_col
        else pd.Series([0.0] * len(df))
    )
    clean_df["Description"] = clean_df["Description"].replace({"nan": "", "None": ""})
    clean_df["Category"] = clean_df["Category"].replace({"nan": "", "None": ""})
    clean_df = clean_df[(clean_df["basictype"] != "") & (clean_df["Name"] != "")].reset_index(
        drop=True
    )

    if len(clean_df) < 10:
        raise ValueError(
            f"[{domain.upper()}] Insufficient training samples ({len(clean_df)} < 10)."
        )

    logger.info(
        f"[{domain.upper()}] Loaded {len(clean_df)} labeled SKUs across "
        f"{clean_df['basictype'].nunique()} unique BasicTypes."
    )
    return clean_df


def extract_embeddings(
    df,
    domain: str,
    cache_dir: str | None = None,
    force_embed: bool = False,
):
    """Extracts or loads cached weighted text embeddings for ArcFace training."""
    import joblib
    import numpy as np
    import pandas as pd

    from engine import config

    cache_dir = cache_dir or config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{domain}_weighted_skus_cache.pkl")

    def _to_clean(series) -> list[str]:
        return [
            str(v).strip() if pd.notna(v) and str(v).lower() not in ("nan", "none") else ""
            for v in series
        ]

    names = _to_clean(df["Name"])
    descs = _to_clean(df["Description"])
    cats = _to_clean(df["Category"])
    keys = [f"{n}||{d}||{c}" for n, d, c in zip(names, descs, cats)]

    sku_cache = {}
    if not force_embed and os.path.exists(cache_file):
        try:
            loaded_cache = joblib.load(cache_file)
            if (
                isinstance(loaded_cache, dict)
                and loaded_cache.get("__embedding_model__") == config.EMBEDDING_MODEL
            ):
                sku_cache = loaded_cache
            elif isinstance(loaded_cache, dict) and loaded_cache:
                logger.info(
                    f"[{domain.upper()}] Weighted SKU cache model mismatch; recomputing for {config.EMBEDDING_MODEL}."
                )
        except Exception:
            sku_cache = {}

    missing_indices = [i for i, k in enumerate(keys) if k not in sku_cache]
    if missing_indices:
        logger.info(
            f"[{domain.upper()}] Embedding {len(missing_indices)} SKUs "
            f"({len(keys) - len(missing_indices)} from cache)..."
        )
        from engine.core.resource_loader import _get_shared_models

        embed_engine, _ = _get_shared_models()
        embs = embed_engine.embed_weighted_sku(
            [names[i] for i in missing_indices],
            [descs[i] for i in missing_indices],
            [cats[i] for i in missing_indices],
            weights=config.CLASSIFIER_WEIGHTS,
        )
        dense_vectors = embs["dense"]
        for idx, i in enumerate(missing_indices):
            sku_cache[keys[i]] = dense_vectors[idx]
        sku_cache["__embedding_model__"] = config.EMBEDDING_MODEL
        joblib.dump(sku_cache, cache_file)
        _record_active_embedding_model()

    return np.vstack([sku_cache[k] for k in keys]).astype(np.float32)


def train_bt_arcface(
    domain: str,
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 1e-3,
    scale: float = 30.0,
    margin: float = 0.35,
    gamma: float = 2.0,
    weight_decay: float = 1e-4,
    device_name: str | None = None,
    from_sample: bool = False,
    sample_file: str = "data/sample/SampleData.xlsx",
    force_embed: bool = False,
    output_dir: str | None = None,
) -> dict[str, str]:
    """Trains BTArcFaceNet, exports ONNX, quantizes INT8, verifies parity, and cleans FP32."""
    import joblib
    import numpy as np
    import onnxruntime as ort
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm

    from engine import config
    from engine.classification.models.arcface_bt import BTArcFaceNet, FocalLoss

    t_start = time.time()
    arcface_dir = output_dir or config.get_arcface_dir()
    os.makedirs(arcface_dir, exist_ok=True)

    df = load_catalog_data(domain, from_sample=from_sample, sample_file=sample_file)
    X_text = extract_embeddings(df, domain, force_embed=force_embed)

    prices = np.clip(df["Price"].values.astype(np.float32).reshape(-1, 1), 0.0, None)
    price_scaler = StandardScaler()
    scaled_prices = price_scaler.fit_transform(np.log1p(prices)).astype(np.float32)

    embed_dim = (
        int(X_text.shape[1]) if X_text.ndim == 2 and X_text.shape[0] > 0 else config.EMBEDDING_DIM
    )
    in_features = embed_dim + 1
    X = np.hstack([X_text, scaled_prices]).astype(np.float32)
    assert X.shape[1] == in_features, f"Expected {in_features}-D features, got {X.shape[1]}"

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["basictype"].tolist())
    num_classes = len(label_encoder.classes_)
    logger.info(f"[{domain.upper()}] ArcFace features={X.shape}, classes={num_classes}")

    device = torch.device(
        device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = BTArcFaceNet(
        in_features=in_features,
        hidden_features=512,
        num_classes=num_classes,
        scale=scale,
        margin=margin,
        dropout=0.2,
    ).to(device)

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long())
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=(len(dataset) > batch_size)
    )
    criterion = FocalLoss(gamma=gamma, reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=1e-5
    )

    model.train()
    epoch_bar = tqdm(range(1, epochs + 1), desc=f"[{domain.upper()}] ArcFace", unit="epoch")
    for epoch in epoch_bar:
        epoch_loss = 0.0
        batches = 0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x, labels=batch_y)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
        scheduler.step()
        avg_loss = epoch_loss / max(batches, 1)
        epoch_bar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")
        if epoch == 1 or epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            logger.info(
                f"[{domain.upper()}] Epoch {epoch:2d}/{epochs:2d} - Focal Loss: {avg_loss:.4f} "
                f"(lr: {scheduler.get_last_lr()[0]:.6f})"
            )

    model.eval()
    model.to("cpu")

    fp32_onnx_path = os.path.join(arcface_dir, f"{domain}_bt_arcface.onnx")
    int8_onnx_path = os.path.join(arcface_dir, f"{domain}_bt_arcface_int8.onnx")
    dummy_input = torch.randn(1, in_features, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        fp32_onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    quantize_model(fp32_onnx_path, int8_onnx_path, f"ARCFACE-{domain.upper()}")

    # Verify INT8 output parity cleanly before deleting intermediate FP32
    sess_int8 = ort.InferenceSession(int8_onnx_path, providers=["CPUExecutionProvider"])
    try:
        probe = X[: min(4, len(X))]
        out_int8 = sess_int8.run(None, {sess_int8.get_inputs()[0].name: probe})[0]
        with torch.no_grad():
            out_pt = model(torch.from_numpy(probe)).numpy()
        if out_int8.shape != out_pt.shape or not np.isfinite(out_int8).all():
            raise RuntimeError(f"[{domain.upper()}] ArcFace INT8 verification failed.")
        max_diff = float(np.max(np.abs(out_int8 - out_pt)))
    except Exception:
        del sess_int8
        gc.collect()
        _remove_onnx_artifacts(int8_onnx_path)
        raise

    del sess_int8
    gc.collect()
    _remove_onnx_artifacts(fp32_onnx_path)
    int8_mb = os.path.getsize(int8_onnx_path) / (1024 * 1024)
    logger.info(
        f"[{domain.upper()}] ArcFace INT8 verified ({int8_mb:.2f} MB, max logit diff={max_diff:.4f}); cleaned FP32."
    )

    labels_json_path = os.path.join(arcface_dir, f"{domain}_bt_arcface_labels.json")
    labels_meta = {
        "domain": domain,
        "num_classes": num_classes,
        "classes": label_encoder.classes_.tolist(),
        "scale": scale,
        "margin": margin,
        "input_dim": in_features,
        "hidden_dim": 512,
        "embedding_model": config.EMBEDDING_MODEL,
        "price_scaler": {
            "mean": float(price_scaler.mean_[0]) if hasattr(price_scaler, "mean_") else 0.0,
            "scale": float(price_scaler.scale_[0]) if hasattr(price_scaler, "scale_") else 1.0,
            "var": float(price_scaler.var_[0]) if hasattr(price_scaler, "var_") else 1.0,
        },
    }
    with open(labels_json_path, "w", encoding="utf-8") as f:
        json.dump(labels_meta, f, indent=2, ensure_ascii=False)

    meta_joblib_path = os.path.join(arcface_dir, f"{domain}_bt_arcface_meta.joblib")
    joblib.dump(
        {
            "label_encoder": label_encoder,
            "price_scaler": price_scaler,
            "classes": label_encoder.classes_.tolist(),
            "scale": scale,
            "margin": margin,
        },
        meta_joblib_path,
    )

    logger.info(
        f"[{domain.upper()}] ArcFace BT completed in {time.time() - t_start:.2f}s -> {int8_onnx_path}"
    )
    return {
        "domain": domain,
        "int8_onnx": int8_onnx_path,
        "labels_json": labels_json_path,
        "meta_joblib": meta_joblib_path,
    }


def sync_arcface(
    domain: str,
    mode: str,
    epochs: int = 25,
    from_sample: bool = False,
    sample_file: str = "data/sample/SampleData.xlsx",
) -> None:
    """Trains ArcFace INT8 head if rebuilding or if missing / mismatched with EMBEDDING_DIM."""
    from engine import config

    _ensure_clean_embedding_caches_for_model()
    onnx_path = config.get_bt_arcface_onnx_path(domain)
    labels_path = config.get_bt_arcface_labels_path(domain)

    needs_train = (
        mode == "rebuild" or not os.path.exists(onnx_path) or not os.path.exists(labels_path)
    )
    if not needs_train:
        labels_meta = _read_manifest(labels_path)
        expected_dim = config.EMBEDDING_DIM + 1
        actual_dim = labels_meta.get("input_dim")
        trained_emb = labels_meta.get("embedding_model")
        if actual_dim != expected_dim or (trained_emb and trained_emb != config.EMBEDDING_MODEL):
            logger.info(
                f"[{domain.upper()}] ArcFace model/dim mismatch "
                f"(trained={trained_emb}/{actual_dim}, active={config.EMBEDDING_MODEL}/{expected_dim}); retraining..."
            )
            needs_train = True

    if not needs_train:
        logger.info(f"[{domain.upper()}] ArcFace INT8 model up-to-date — skipped.")
        return

    train_bt_arcface(
        domain=domain,
        epochs=epochs,
        from_sample=from_sample,
        sample_file=sample_file,
        force_embed=(mode == "rebuild"),
    )


# ---------------------------------------------------------------------------
# 4. Orchestration & CLI Entrypoint
# ---------------------------------------------------------------------------


def resolve_stage_modes(args: argparse.Namespace) -> dict[str, str | None]:
    """Resolves effective modes across models, db, cache, qdrant, and arcface."""
    explicit_any = any(
        getattr(args, name) is not None for name in ("models", "db", "cache", "qdrant", "arcface")
    )
    if explicit_any:
        return {
            "models": "rebuild" if (args.rebuild and args.models is None) else args.models,
            "db": "rebuild" if (args.rebuild and args.db is None) else args.db,
            "cache": "rebuild" if (args.rebuild and args.cache is None) else args.cache,
            "qdrant": "rebuild" if (args.rebuild and args.qdrant is None) else args.qdrant,
            "arcface": "rebuild" if (args.rebuild and args.arcface is None) else args.arcface,
        }
    if args.from_sample or args.rebuild:
        return dict.fromkeys(("models", "db", "cache", "qdrant", "arcface"), "rebuild")
    return dict.fromkeys(("models", "db", "cache", "qdrant", "arcface"), "sync")


def _get_sheet_id(need_engine: bool) -> str | None:
    if need_engine:
        from engine import config

        return config.GOOGLE_SHEET_ID
    try:
        from backend.app.core import config as backend_config

        return backend_config.GOOGLE_SHEET_ID
    except Exception:
        return os.getenv("GOOGLE_SHEET_ID")


def run(
    models_mode: str | None = None,
    model_target: str = "all",
    db_mode: str | None = None,
    cache_mode: str | None = None,
    qdrant_mode: str | None = None,
    arcface_mode: str | None = None,
    target_domains: list[str] | None = None,
    from_staged: bool = False,
    from_sample: bool = False,
    sample_file: str = "data/sample/SampleData.xlsx",
    keep_staged: bool = False,
    epochs: int = 25,
    is_quantize_sub: bool = False,
) -> None:
    if is_quantize_sub:
        sync_models(mode=models_mode or "quantize", target=model_target, _is_quantize_sub=True)
        return

    if target_domains is None:
        target_domains = [_DOMAIN_MARKET, _DOMAIN_FOOD]

    sample_file = _resolve_project_path(sample_file)

    logger.info("=" * 60)
    logger.info("SKU MATCHOPS UNIFIED SYNC")
    logger.info(
        f"--models={models_mode or 'skip'} ({model_target})  "
        f"--db={db_mode or 'skip'}  --cache={cache_mode or 'skip'}  "
        f"--qdrant={qdrant_mode or 'skip'}  --arcface={arcface_mode or 'skip'}"
    )
    logger.info(f"Domains: {[d.upper() for d in target_domains]}")
    logger.info("=" * 60)

    if models_mode:
        sync_models(mode=models_mode, target=model_target)

    need_catalog_data = any((db_mode, cache_mode, qdrant_mode, arcface_mode))
    if not need_catalog_data:
        logger.info("MODEL STEP COMPLETE.")
        return

    need_engine = bool(models_mode or cache_mode or qdrant_mode or arcface_mode)
    sheet_id = "SAMPLE_WORKBOOK" if from_sample else _get_sheet_id(need_engine)
    if not from_sample and not sheet_id and db_mode and not from_staged:
        logger.error(
            "GOOGLE_SHEET_ID is not configured in .env (or use --sample for offline demo mode)."
        )
        sys.exit(1)

    single_domain = target_domains[0] if len(target_domains) == 1 else None
    if from_sample:
        if cache_mode:
            if single_domain:
                wipe_domain_cache_files(single_domain)
            else:
                wipe_local_cache_files()
        if db_mode:
            reset_catalog_tables(single_domain)
        logger.info(f"\n>>> EXTRACTING SAMPLE DATA FROM '{sample_file}' <<<")
        if not os.path.exists(sample_file):
            logger.critical(f"Sample data file not found at '{sample_file}'.")
            sys.exit(1)
        _stage_all_from_excel(sample_file, domains=target_domains)
    elif db_mode and not from_staged:
        logger.info("\n>>> PRE-FETCHING & STAGING GOOGLE SHEETS <<<")
        _stage_all_sheets(sheet_id, domains=target_domains)
        logger.info("Staged locally — remaining pipeline runs 100% offline.")
    elif from_staged:
        logger.info("\n>>> USING PREVIOUSLY STAGED GOOGLE SHEETS (OFFLINE MODE) <<<")

    for domain in target_domains:
        logger.info(f"\n>>> DOMAIN: {domain.upper()} <<<")

        if db_mode:
            logger.info(f"[{domain.upper()}] --db ({db_mode})...")
            sync_db(domain, sheet_id, db_mode)

        if cache_mode:
            logger.info(f"[{domain.upper()}] --cache ({cache_mode})...")
            if cache_mode == "rebuild":
                wipe_domain_cache_files(domain)
            sync_cache(domain, sheet_id, cache_mode)

        if qdrant_mode:
            logger.info(f"[{domain.upper()}] --qdrant ({qdrant_mode})...")
            sync_qdrant(domain, qdrant_mode)

        if arcface_mode:
            logger.info(f"[{domain.upper()}] --arcface ({arcface_mode})...")
            sync_arcface(
                domain,
                arcface_mode,
                epochs=epochs,
                from_sample=from_sample,
                sample_file=sample_file,
            )

    if not keep_staged:
        logger.info("\n>>> CLEANING UP TEMPORARY STAGED SHEETS <<<")
        _cleanup_staged_sheets()

    logger.info("\n" + "=" * 60)
    logger.info("SYNC COMPLETE.")
    logger.info("=" * 60)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SKU MatchOps Unified CLI (--models / --db / --cache / --qdrant / --arcface).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models",
        choices=["sync", "export", "quantize", "rebuild"],
        default=None,
        help="ONNX model export, INT8 quantization, and automatic FP32-vs-INT8 verification.",
    )
    parser.add_argument(
        "--model-target",
        choices=["all", "embedding", "reranker", "gliner"],
        default="all",
        help="Scope for --models (default: all).",
    )
    parser.add_argument(
        "--db",
        choices=["sync", "rebuild"],
        default=None,
        help="PostgreSQL catalog tables.",
    )
    parser.add_argument(
        "--cache",
        choices=["sync", "rebuild"],
        default=None,
        help="Local disk caches (Feather, dictionaries, NLP metadata).",
    )
    parser.add_argument(
        "--qdrant",
        choices=["sync", "rebuild"],
        default=None,
        help="Qdrant hybrid vectors & classifier models.",
    )
    parser.add_argument(
        "--arcface",
        choices=["sync", "rebuild"],
        default=None,
        help="ArcFace BasicType metric head training & INT8 export.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Shortcut to rebuild all unpassed stages (models, db, cache, qdrant, arcface).",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="all",
        choices=["all", "market", "food"],
        help="Target domain (default: all).",
    )
    parser.add_argument(
        "--sample",
        dest="from_sample",
        action="store_true",
        help="Offline demo mode from data/sample/SampleData.xlsx.",
    )
    parser.add_argument(
        "--sample-file",
        dest="sample_file",
        type=str,
        default="data/sample/SampleData.xlsx",
        help="Path to sample Excel workbook.",
    )
    parser.add_argument(
        "--from-staged",
        "--offline",
        dest="from_staged",
        action="store_true",
        help="Use previously staged sheets instead of downloading from Google Sheets.",
    )
    parser.add_argument(
        "--keep-staged",
        dest="keep_staged",
        action="store_true",
        help="Keep temporary staged CSV files after sync finishes.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Training epochs for --arcface (default: 25).",
    )
    parser.add_argument(
        "--_quantize-sub",
        dest="is_quantize_sub",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    modes = resolve_stage_modes(args)

    if args.domain == "market":
        target_domains = [_DOMAIN_MARKET]
    elif args.domain == "food":
        target_domains = [_DOMAIN_FOOD]
    else:
        target_domains = [_DOMAIN_MARKET, _DOMAIN_FOOD]

    run(
        models_mode=modes["models"],
        model_target=args.model_target,
        db_mode=modes["db"],
        cache_mode=modes["cache"],
        qdrant_mode=modes["qdrant"],
        arcface_mode=modes["arcface"],
        target_domains=target_domains,
        from_staged=args.from_staged,
        from_sample=args.from_sample,
        sample_file=args.sample_file,
        keep_staged=args.keep_staged,
        epochs=args.epochs,
        is_quantize_sub=args.is_quantize_sub,
    )


if __name__ == "__main__":
    main()
