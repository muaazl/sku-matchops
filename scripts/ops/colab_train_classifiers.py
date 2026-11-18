"""
Google Colab GPU Classifier & ArcFace Trainer for SKU MatchOps.

Downloads Google Sheets (or reads data/staged_sheets/), computes dense SKU &
dictionary embeddings on GPU, trains sklearn classifiers (BT / Third-Tag / GK) +
ColdStartRouter warm centroids, trains the ArcFace BT head on GPU and exports
INT8 ONNX, and packages all artifacts into a single zip ready to extract into
your local project root.

Usage (in Colab or locally with GPU):
    python -m scripts.ops.colab_train_classifiers --sheet-id <GOOGLE_SHEET_ID> --domain food
    python -m scripts.ops.colab_train_classifiers --sheet-id <GOOGLE_SHEET_ID> --domain all
    python -m scripts.ops.colab_train_classifiers --from-staged --domain market
"""

import argparse
import logging
import os
import sys
import threading
import time
import zipfile
from collections import OrderedDict

import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Prevent ZeroShotClassifier.__init__ from failing fast on missing ArcFace ONNX
# before ArcFace has been trained in this run.
os.environ.setdefault("FOOD_BT_MODEL", "logreg")
os.environ.setdefault("MARKET_BT_MODEL", "logreg")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("matchops.colab_train")


def _load_gpu_embedding_engine(batch_size: int):
    """Builds a CUDA-accelerated EmbeddingEngine compatible with ZeroShotClassifier & sync.py."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    from engine import config
    from engine.nlp.embedding_engine import EmbeddingEngine
    from scripts.sync import (
        EmbeddingONNXWrapper,
        _detect_pooling_mode,
        _load_sparse_linear,
        _resolve_hf_embedding_id,
    )

    model_id = config.EMBEDDING_MODEL
    hf_id = _resolve_hf_embedding_id(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    logger.info(
        f"[GPU-EMBED] Loading '{hf_id}' on {device} (dtype={dtype}, batch_size={batch_size})..."
    )
    hf_config = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
    pooling = _detect_pooling_mode("", hf_id, model_id, hf_config)
    hidden_size = int(getattr(hf_config, "hidden_size", 1024))
    sparse_linear = _load_sparse_linear("", hf_id, model_id, hidden_size)

    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True, fix_mistral_regex=True)
    tokenizer.padding_side = "right"

    hf_model = AutoModel.from_pretrained(hf_id, trust_remote_code=True, torch_dtype=dtype)
    hf_model.eval()
    wrapper = EmbeddingONNXWrapper(hf_model, pooling=pooling, sparse_linear=sparse_linear).to(
        device
    )
    wrapper.eval()

    os.environ["EMBEDDING_DIM"] = str(hidden_size)
    config.refresh_model_config()

    class GpuEmbeddingEngine(EmbeddingEngine):
        """Drop-in EmbeddingEngine replacement running PyTorch on GPU/CPU."""

        def __init__(self):
            self._cache_lock = threading.RLock()
            self._str_cache: OrderedDict[
                str, tuple[np.ndarray, dict[int, float], dict[int, float] | None]
            ] = OrderedDict()
            self.tokenizer = tokenizer
            self.wrapper = wrapper
            self.device = device
            self.batch_size = batch_size
            self._emb_pooling = pooling
            self.embedding_dim = hidden_size
            self.has_learned_sparse = sparse_linear is not None
            self.bi_session = None
            self.cross_session = None

        def _load_models(self):
            pass

        def encode(
            self, texts: list[str], batch_size: int | None = None
        ) -> dict[str, np.ndarray | list[dict]]:
            if not texts:
                return {"dense": np.array([]), "bm25": [], "sparse": []}

            eff_batch = batch_size or self.batch_size
            resolved: dict[str, tuple[np.ndarray, dict[int, float], dict[int, float] | None]] = {}
            missing_texts: list[str] = []
            missing_set: set[str] = set()

            with self._cache_lock:
                for text in texts:
                    if not text or not str(text).strip():
                        continue
                    if text in self._str_cache:
                        self._str_cache.move_to_end(text)
                        resolved[text] = self._str_cache[text]
                    elif text not in missing_set:
                        missing_texts.append(text)
                        missing_set.add(text)

            unused_tokens = set(getattr(self.tokenizer, "all_special_ids", ()))

            if missing_texts:
                all_dense: list[np.ndarray] = []
                all_bm25: list[dict[int, float]] = []
                all_learned_sparse: list[dict[int, float] | None] = []

                starts = range(0, len(missing_texts), eff_batch)
                if len(missing_texts) > eff_batch:
                    starts = tqdm(
                        starts,
                        desc=f"[GPU-EMBED] Encoding ({len(missing_texts)} texts)",
                        unit="batch",
                        mininterval=0.5,
                        leave=False,
                    )

                with torch.inference_mode():
                    for i in starts:
                        batch = missing_texts[i : i + eff_batch]
                        inputs = self.tokenizer(
                            batch,
                            padding=True,
                            truncation=True,
                            max_length=512,
                            return_tensors="pt",
                        )
                        input_ids = inputs["input_ids"].to(self.device)
                        attention_mask = inputs["attention_mask"].to(self.device)

                        out = self.wrapper(input_ids, attention_mask)
                        if isinstance(out, tuple):
                            dense_t, sparse_t = out
                            sparse_np = sparse_t.float().cpu().numpy()
                        else:
                            dense_t = out
                            sparse_np = None

                        dense_np = dense_t.float().cpu().numpy()
                        norms = np.linalg.norm(dense_np, axis=1, keepdims=True)
                        norms = np.where(norms == 0, 1e-12, norms)
                        all_dense.append(dense_np / norms)

                        for text_item in batch:
                            all_bm25.append(self.compute_bm25_tf(text_item))

                        if sparse_np is not None:
                            ids_np = inputs["input_ids"].numpy()
                            for b in range(len(batch)):
                                d: dict[int, float] = {}
                                for tok_id, w in zip(ids_np[b], sparse_np[b]):
                                    tok_id = int(tok_id)
                                    if tok_id not in unused_tokens and w > 0:
                                        d[tok_id] = max(d.get(tok_id, 0.0), float(w))
                                all_learned_sparse.append(d)
                        else:
                            all_learned_sparse.extend([None] * len(batch))

                flat_dense = np.vstack(all_dense) if all_dense else np.array([])
                max_cache_size = int(os.getenv("STR_CACHE_MAX_SIZE", "25000"))
                with self._cache_lock:
                    for i, text in enumerate(missing_texts):
                        entry = (flat_dense[i], all_bm25[i], all_learned_sparse[i])
                        resolved[text] = entry
                        self._str_cache[text] = entry
                        self._str_cache.move_to_end(text)
                    while len(self._str_cache) > max_cache_size:
                        self._str_cache.popitem(last=False)

            first_non_empty = next(iter(resolved.values()), None)
            dim = first_non_empty[0].shape[0] if first_non_empty is not None else self.embedding_dim
            zero_dense = np.zeros(dim, dtype=np.float32)
            out_dense = np.empty((len(texts), dim), dtype=np.float32)
            out_bm25: list[dict[int, float]] = []
            out_sparse: list[dict[int, float]] = []

            for i, text in enumerate(texts):
                if not text or not str(text).strip() or text not in resolved:
                    out_dense[i] = zero_dense
                    out_bm25.append({})
                    if self.has_learned_sparse:
                        out_sparse.append({})
                    continue
                d, bm25_d, learned_d = resolved[text]
                out_dense[i] = d
                out_bm25.append(bm25_d)
                if self.has_learned_sparse:
                    out_sparse.append(learned_d if learned_d is not None else {})

            return {"dense": out_dense, "bm25": out_bm25, "sparse": out_sparse}

    return GpuEmbeddingEngine()


def _train_domain_sklearn_and_caches(
    domain: str,
    sheet_id: str,
    gpu_embedder,
) -> None:
    """Builds Feather caches, BT-GK map, tag incremental caches, sklearn classifiers & centroids."""
    import json

    from engine import config
    from engine.classification.classifier import ZeroShotClassifier
    from engine.classification.loader import (
        augment_bt_gk_map_with_training,
        build_bt_third_tag_map_from_catalog,
        build_descriptions,
        build_umbrella_from_training,
    )
    from engine.data_pipeline.cache_manager import CacheManager
    from engine.data_pipeline.ingestion import DataIngestion
    from engine.nlp.ner_engine import NEREngine

    os.makedirs(config.CACHE_DIR, exist_ok=True)
    DataIngestion.clear_mem_cache(domain)
    cat_df, brands_df = DataIngestion.load_catalog(sheet_id, domain=domain, force_fetch=True)
    dicts = DataIngestion.load_classifier_dictionaries(sheet_id, domain=domain, force_fetch=True)

    dicts_cache_path = os.path.join(config.CACHE_DIR, f"{domain}_classifier_dicts.json")
    with open(dicts_cache_path, "w", encoding="utf-8") as f:
        json.dump(dicts, f, indent=2)

    # NEREngine without loading GLiNER ONNX (only used as lightweight wrapper here)
    ner_stub = NEREngine.__new__(NEREngine)
    ner_stub.labels = (
        config.FOOD_NER_LABELS if domain == config.DOMAIN_FOOD else config.MARKET_NER_LABELS
    )
    ner_stub.target_label = "flavor" if domain == config.DOMAIN_FOOD else "brand"
    ner_stub.model = None
    ner_stub.brand_mapping = {}

    cache_manager = CacheManager(ner_stub, gpu_embedder)

    def _build_bt_gk(df):
        bt_map = augment_bt_gk_map_with_training(df, {}, dicts.get("gk", []))
        umbrella = build_umbrella_from_training(df)
        third_tag_map = build_bt_third_tag_map_from_catalog(df, domain)
        return bt_map, umbrella, third_tag_map

    bt_gk_cache = cache_manager.get_or_build_bt_gk_cache(cat_df, domain, _build_bt_gk)
    descriptions = build_descriptions(
        domain,
        dicts,
        bt_gk_cache["bt_gk_map"],
        bt_gk_cache.get("third_tag_map", {}),
    )
    descriptions["bt_to_gk_umbrella"] = bt_gk_cache["umbrella"]

    third_tag_key = "region" if domain == config.DOMAIN_FOOD else "category"
    for tag_key, dict_type in (
        ("gk", "gk"),
        ("bt", "bt"),
        (third_tag_key, third_tag_key),
    ):
        tag_list = dicts.get(tag_key, [])
        if tag_list:
            gpu_embedder.embed_dictionary_incremental(domain, dict_type, tag_list)

    clf = ZeroShotClassifier(
        gpu_embedder,
        domain,
        descriptions,
        cat_df=cat_df,
        brands_df=brands_df,
        force_retrain=True,
    )
    if not clf._trained:
        raise RuntimeError(f"[{domain.upper()}] ZeroShotClassifier training failed.")


def _package_artifacts(domains: list[str], output_zip: str) -> str:
    """Zips generated cache and ArcFace artifacts with repo-relative paths."""
    from engine import config

    output_path = (
        output_zip if os.path.isabs(output_zip) else os.path.join(PROJECT_ROOT, output_zip)
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    arcface_dir = config.get_arcface_dir()
    files_to_pack: list[str] = []

    shared_cache = ("active_embedding_model.txt", "model_hashes.json")
    for fname in shared_cache:
        fpath = os.path.join(config.CACHE_DIR, fname)
        if os.path.isfile(fpath):
            files_to_pack.append(fpath)

    if os.path.isdir(config.CACHE_DIR):
        for fname in sorted(os.listdir(config.CACHE_DIR)):
            fpath = os.path.join(config.CACHE_DIR, fname)
            if not os.path.isfile(fpath):
                continue
            if any(fname.startswith(f"{d}_") for d in domains):
                files_to_pack.append(fpath)

    if os.path.isdir(arcface_dir):
        for fname in sorted(os.listdir(arcface_dir)):
            fpath = os.path.join(arcface_dir, fname)
            if not os.path.isfile(fpath):
                continue
            if any(fname.startswith(f"{d}_") for d in domains):
                files_to_pack.append(fpath)

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fpath in files_to_pack:
            arcname = os.path.relpath(fpath, PROJECT_ROOT).replace("\\", "/")
            zf.write(fpath, arcname=arcname)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"[PACK] Wrote {len(files_to_pack)} artifacts to {output_path} ({size_mb:.2f} MB)")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train SKU MatchOps classifiers & ArcFace head on Colab GPU."
    )
    parser.add_argument(
        "--sheet-id",
        type=str,
        default=os.getenv("GOOGLE_SHEET_ID", ""),
        help="Google Sheet ID (defaults to GOOGLE_SHEET_ID env/.env).",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="all",
        choices=["all", "market", "food"],
        help="Target domain (default: all).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="ArcFace training epochs (default: 25).",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=256,
        help="Batch size for GPU embedding generation (default: 256).",
    )
    parser.add_argument(
        "--arcface-batch-size",
        type=int,
        default=64,
        help="Batch size for ArcFace training (default: 64).",
    )
    parser.add_argument(
        "--output-zip",
        type=str,
        default="matchops_colab_artifacts.zip",
        help="Output zip file path (default: matchops_colab_artifacts.zip).",
    )
    parser.add_argument(
        "--from-staged",
        action="store_true",
        help="Use existing CSVs in data/staged_sheets/ instead of downloading Google Sheets.",
    )
    parser.add_argument(
        "--keep-staged",
        action="store_true",
        help="Keep data/staged_sheets/ after training finishes.",
    )
    parser.add_argument(
        "--skip-sklearn",
        action="store_true",
        help="Skip sklearn ZeroShotClassifier & centroid training.",
    )
    parser.add_argument(
        "--skip-arcface",
        action="store_true",
        help="Skip ArcFace PyTorch training & ONNX INT8 export.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="",
        help="Optional override for EMBEDDING_MODEL (defaults to .env / config).",
    )
    args = parser.parse_args()

    domains = ["market", "food"] if args.domain == "all" else [args.domain]
    if not args.from_staged and not args.sheet_id:
        parser.error("Provide --sheet-id <GOOGLE_SHEET_ID> or set GOOGLE_SHEET_ID / --from-staged.")

    if args.sheet_id:
        os.environ["GOOGLE_SHEET_ID"] = args.sheet_id
    if args.embedding_model:
        os.environ["EMBEDDING_MODEL"] = args.embedding_model

    t0 = time.time()
    from engine import config
    from engine.core import resource_loader
    from scripts import sync as sync_mod

    if args.sheet_id:
        config.GOOGLE_SHEET_ID = args.sheet_id
    if args.embedding_model:
        config.refresh_model_config()

    if not args.from_staged:
        logger.info(f"[STAGE] Downloading Google Sheets for {domains}...")
        sync_mod._stage_all_sheets(args.sheet_id, domains=domains)

    gpu_embedder = _load_gpu_embedding_engine(batch_size=args.embed_batch_size)
    # Patch _get_shared_models so train_bt_arcface reuses GPU embeddings too
    resource_loader._get_shared_models = lambda: (gpu_embedder, None)

    for domain in domains:
        logger.info(f"\n{'=' * 60}\nTRAINING DOMAIN: {domain.upper()}\n{'=' * 60}")
        if not args.skip_sklearn:
            _train_domain_sklearn_and_caches(domain, args.sheet_id or "STAGED", gpu_embedder)

        if not args.skip_arcface:
            sync_mod.train_bt_arcface(
                domain=domain,
                epochs=args.epochs,
                batch_size=args.arcface_batch_size,
                device_name="cuda" if torch.cuda.is_available() else "cpu",
                from_sample=False,
                force_embed=False,
            )

    sync_mod._record_active_embedding_model()
    zip_path = _package_artifacts(domains, args.output_zip)

    if not args.keep_staged and not args.from_staged:
        sync_mod._cleanup_staged_sheets()

    logger.info(
        f"\nAll done in {time.time() - t0:.1f}s! Extract '{zip_path}' into your local repo root, "
        f"then run locally: python -m scripts.sync --qdrant sync"
    )


if __name__ == "__main__":
    main()
