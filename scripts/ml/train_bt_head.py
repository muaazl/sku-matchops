"""
SKU MatchOps - Offline ArcFace BasicType (BT) Head Training & Export CLI

Trains a deep metric learning PyTorch ArcFace model (BTArcFaceNet) on SKU catalog
embeddings and prices, exports the evaluation graph to ONNX, quantizes to INT8,
and serializes label mappings and price scalers.

Usage:
  python -m scripts.ml.train_bt_head --domain food
  python -m scripts.ml.train_bt_head --domain market
  python -m scripts.ml.train_bt_head --domain all
  python -m scripts.ml.train_bt_head --domain market --from-sample
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

# Resolve project roots
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(ENGINE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine import config
from engine.classification.models.arcface_bt import BTArcFaceNet, FocalLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("matchops.train_bt_head")


def load_catalog_data(
    domain: str,
    from_sample: bool = False,
    sample_file: str = "data/sample/SampleData.xlsx",
) -> pd.DataFrame:
    """Loads and standardizes catalog data for a given domain."""
    df = pd.DataFrame()

    if from_sample:
        if not os.path.exists(sample_file):
            raise FileNotFoundError(f"Sample workbook not found: {sample_file}")
        sheet_name = (
            config.FOOD_CATALOG_SHEET
            if domain == config.DOMAIN_FOOD
            else config.MARKET_CATALOG_SHEET
        )
        logger.info(f"[{domain.upper()}] Loading training data from sample sheet '{sheet_name}' in {sample_file}...")
        xl = pd.ExcelFile(sample_file)
        if sheet_name in xl.sheet_names:
            df = xl.parse(sheet_name)
    else:
        try:
            from engine.data_pipeline.ingestion import DataIngestion
            cat_df, _ = DataIngestion.load_catalog(config.GOOGLE_SHEET_ID, domain=domain)
            if not cat_df.empty:
                df = cat_df
        except Exception as e:
            logger.warning(f"[{domain.upper()}] Database/Sheet load failed: {e}. Checking sample fallback...")

        if df.empty and os.path.exists(sample_file):
            logger.info(f"[{domain.upper()}] Loading catalog fallback from {sample_file}...")
            sheet_name = (
                config.FOOD_CATALOG_SHEET
                if domain == config.DOMAIN_FOOD
                else config.MARKET_CATALOG_SHEET
            )
            xl = pd.ExcelFile(sample_file)
            if sheet_name in xl.sheet_names:
                df = xl.parse(sheet_name)

    if df.empty:
        raise ValueError(f"[{domain.upper()}] No catalog data found for training.")

    # Column standardization
    col_map = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    
    # Detect BasicType column
    bt_col = None
    for cand in ["basictype", "bt"]:
        if cand in col_map:
            bt_col = col_map[cand]
            break
    if not bt_col:
        raise ValueError(f"[{domain.upper()}] Catalog missing BasicType column in: {list(df.columns)}")

    # Detect Name column
    name_col = None
    for cand in ["name", "skuname", "itemname"]:
        if cand in col_map:
            name_col = col_map[cand]
            break
    if not name_col:
        raise ValueError(f"[{domain.upper()}] Catalog missing Name column in: {list(df.columns)}")

    # Detect Description column
    desc_col = None
    for cand in ["description", "desc"]:
        if cand in col_map:
            desc_col = col_map[cand]
            break

    # Detect Category / Region column
    cat_col = None
    for cand in ["category", "cat", "region"]:
        if cand in col_map:
            cat_col = col_map[cand]
            break

    # Detect Price column
    price_col = None
    for cand in ["price"]:
        if cand in col_map:
            price_col = col_map[cand]
            break

    df = df.fillna("")
    clean_df = pd.DataFrame()
    clean_df["basictype"] = df[bt_col].astype(str).str.strip()
    clean_df["Name"] = df[name_col].astype(str).str.strip()
    clean_df["Description"] = (
        df[desc_col].astype(str).str.strip()
        if desc_col and desc_col in df.columns
        else pd.Series([""] * len(df))
    )
    clean_df["Category"] = (
        df[cat_col].astype(str).str.strip()
        if cat_col and cat_col in df.columns
        else pd.Series([""] * len(df))
    )
    clean_df["Price"] = (
        pd.to_numeric(df[price_col], errors="coerce").fillna(0.0)
        if price_col and price_col in df.columns
        else pd.Series([0.0] * len(df))
    )

    # Clean string literals representing missing values
    clean_df["Description"] = clean_df["Description"].replace({"nan": "", "None": ""})
    clean_df["Category"] = clean_df["Category"].replace({"nan": "", "None": ""})

    # Filter invalid rows
    clean_df = clean_df[(clean_df["basictype"] != "") & (clean_df["Name"] != "")]
    clean_df = clean_df.reset_index(drop=True)

    if len(clean_df) < 10:
        raise ValueError(f"[{domain.upper()}] Insufficient training samples ({len(clean_df)} < 10).")

    logger.info(
        f"[{domain.upper()}] Loaded {len(clean_df)} labeled SKUs across "
        f"{clean_df['basictype'].nunique()} unique BasicTypes."
    )
    return clean_df


def extract_embeddings(
    df: pd.DataFrame,
    domain: str,
    cache_dir: Optional[str] = None,
    force_embed: bool = False,
) -> np.ndarray:
    """Extracts or loads cached multi-field weighted text embeddings."""
    cache_dir = cache_dir or config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{domain}_weighted_skus_cache.pkl")

    def _to_clean_str_list(series) -> List[str]:
        return [
            str(v).strip() if pd.notna(v) and str(v).lower() not in ("nan", "none") else ""
            for v in series
        ]

    names = _to_clean_str_list(df["Name"])
    descs = _to_clean_str_list(df["Description"])
    cats = _to_clean_str_list(df["Category"])
    keys = [f"{n}||{d}||{c}" for n, d, c in zip(names, descs, cats)]

    sku_cache = {}
    if not force_embed and os.path.exists(cache_file):
        try:
            sku_cache = joblib.load(cache_file)
        except Exception:
            sku_cache = {}

    missing_indices = [i for i, k in enumerate(keys) if k not in sku_cache]

    if missing_indices:
        logger.info(f"[{domain.upper()}] Embedding {len(missing_indices)} SKUs ({len(keys) - len(missing_indices)} from cache)...")
        try:
            from engine.core.resource_loader import _get_shared_models
            embed_engine, _ = _get_shared_models()
            missing_names = [names[i] for i in missing_indices]
            missing_descs = [descs[i] for i in missing_indices]
            missing_cats = [cats[i] for i in missing_indices]

            embs = embed_engine.embed_weighted_sku(
                missing_names, missing_descs, missing_cats, weights=config.CLASSIFIER_WEIGHTS
            )
            dense_vectors = embs["dense"]
            for idx, i in enumerate(missing_indices):
                sku_cache[keys[i]] = dense_vectors[idx]

            joblib.dump(sku_cache, cache_file)
        except Exception as e:
            logger.warning(f"[{domain.upper()}] Online embed_engine failed ({e}). Checking fallback cache...")
            if not sku_cache:
                # Synthetic fallback for mock/offline testing environments
                logger.warning(f"[{domain.upper()}] Generating deterministic pseudo-embeddings for {len(keys)} items...")
                for i, k in enumerate(keys):
                    rng = np.random.RandomState(abs(hash(k)) % (2**32))
                    sku_cache[k] = rng.randn(1024).astype(np.float32)

    X_dense = np.vstack([sku_cache[k] for k in keys]).astype(np.float32)
    return X_dense


def train_bt_arcface(
    domain: str,
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 1e-3,
    scale: float = 30.0,
    margin: float = 0.35,
    gamma: float = 2.0,
    weight_decay: float = 1e-4,
    device_name: Optional[str] = None,
    from_sample: bool = False,
    sample_file: str = "data/sample/SampleData.xlsx",
    force_embed: bool = False,
    no_quantize: bool = False,
    output_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Trains BTArcFaceNet, exports to ONNX, quantizes to INT8, and saves label mapping.
    """
    t_start = time.time()
    if output_dir:
        arcface_dir = output_dir
    else:
        arcface_dir = config.get_arcface_dir() if hasattr(config, "get_arcface_dir") else getattr(config, "ARCFACE_DIR", config.ONNX_DIR)
    os.makedirs(arcface_dir, exist_ok=True)

    # 1. Load data
    df = load_catalog_data(domain, from_sample=from_sample, sample_file=sample_file)

    # 2. Extract multi-field dense text embeddings (1024-D)
    X_text = extract_embeddings(df, domain, force_embed=force_embed)

    # 3. Preprocess log prices (1-D)
    prices = np.clip(df["Price"].values.astype(np.float32).reshape(-1, 1), 0.0, None)
    log_prices = np.log1p(prices)
    price_scaler = StandardScaler()
    scaled_prices = price_scaler.fit_transform(log_prices).astype(np.float32)

    # 4. Fuse features -> 1025-D
    X = np.hstack([X_text, scaled_prices]).astype(np.float32)
    assert X.shape[1] == 1025, f"Expected 1025-D features, got {X.shape[1]}"

    # 5. Encode BT target labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["basictype"].tolist())
    num_classes = len(label_encoder.classes_)
    logger.info(f"[{domain.upper()}] Features shape: {X.shape}, Classes: {num_classes}")

    # 6. Setup PyTorch training
    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"[{domain.upper()}] Training device: {device}")

    model = BTArcFaceNet(
        in_features=1025,
        hidden_features=512,
        num_classes=num_classes,
        scale=scale,
        margin=margin,
        dropout=0.2,
    ).to(device)

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=(len(dataset) > batch_size))

    criterion = FocalLoss(gamma=gamma, reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-5)

    model.train()
    for epoch in range(1, epochs + 1):
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
        if epoch == 1 or epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            logger.info(f"[{domain.upper()}] Epoch {epoch:2d}/{epochs:2d} - Focal Loss: {avg_loss:.4f} (lr: {scheduler.get_last_lr()[0]:.6f})")

    # 7. Export FP32 ONNX evaluation graph (inference forward without margins)
    model.eval()
    model.to("cpu")

    fp32_onnx_path = os.path.join(arcface_dir, f"{domain}_bt_arcface.onnx")
    dummy_input = torch.randn(1, 1025, dtype=torch.float32)

    logger.info(f"[{domain.upper()}] Exporting evaluation graph to ONNX: {fp32_onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        fp32_onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    fp32_size_mb = os.path.getsize(fp32_onnx_path) / (1024 * 1024)
    logger.info(f"[{domain.upper()}] FP32 ONNX export complete ({fp32_size_mb:.2f} MB).")

    # 8. Dynamic INT8 Quantization
    int8_onnx_path = os.path.join(arcface_dir, f"{domain}_bt_arcface_int8.onnx")
    if not no_quantize:
        logger.info(f"[{domain.upper()}] Quantizing model to INT8 via onnxruntime...")
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(
            model_input=fp32_onnx_path,
            model_output=int8_onnx_path,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
        )
        int8_size_mb = os.path.getsize(int8_onnx_path) / (1024 * 1024)
        logger.info(f"[{domain.upper()}] INT8 ONNX quantized successfully ({int8_size_mb:.2f} MB).")

    # 9. Serialize Label Mapping and Scaler Parameters
    labels_json_path = os.path.join(arcface_dir, f"{domain}_bt_arcface_labels.json")
    labels_meta = {
        "domain": domain,
        "num_classes": num_classes,
        "classes": label_encoder.classes_.tolist(),
        "scale": scale,
        "margin": margin,
        "input_dim": 1025,
        "hidden_dim": 512,
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

    elapsed = time.time() - t_start
    logger.info(
        f"[{domain.upper()}] [OK] ArcFace BT pipeline completed in {elapsed:.2f}s! Artifacts:\n"
        f"  - FP32 ONNX: {fp32_onnx_path}\n"
        f"  - INT8 ONNX: {int8_onnx_path}\n"
        f"  - Labels JSON: {labels_json_path}\n"
        f"  - Meta Joblib: {meta_joblib_path}"
    )

    return {
        "domain": domain,
        "fp32_onnx": fp32_onnx_path,
        "int8_onnx": int8_onnx_path,
        "labels_json": labels_json_path,
        "meta_joblib": meta_joblib_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="SKU MatchOps ArcFace BasicType Head Offline Trainer & INT8 Exporter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="all",
        choices=["food", "market", "all"],
        help="Catalog domain to train and export.",
    )
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate for AdamW.")
    parser.add_argument("--scale", type=float, default=30.0, help="ArcFace angular scale s.")
    parser.add_argument("--margin", type=float, default=0.35, help="ArcFace additive angular margin m.")
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma parameter.")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', 'cuda', etc.).")
    parser.add_argument("--from-sample", action="store_true", help="Force loading catalog from SampleData.xlsx.")
    parser.add_argument(
        "--sample-file",
        type=str,
        default="data/sample/SampleData.xlsx",
        help="Path to sample Excel workbook.",
    )
    parser.add_argument("--force-embed", action="store_true", help="Re-compute text embeddings instead of reading cache.")
    parser.add_argument("--no-quantize", action="store_true", help="Skip INT8 dynamic quantization.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save exported ONNX and labels (defaults to config.ARCFACE_DIR).")

    args = parser.parse_args()

    domains = (
        [config.DOMAIN_FOOD, config.DOMAIN_MARKET]
        if args.domain == "all"
        else [args.domain.lower()]
    )

    for domain in domains:
        logger.info(f"\n{'=' * 60}\nTRAINING ARCFACE BT HEAD: {domain.upper()}\n{'=' * 60}")
        train_bt_arcface(
            domain=domain,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            scale=args.scale,
            margin=args.margin,
            gamma=args.gamma,
            device_name=args.device,
            from_sample=args.from_sample,
            sample_file=args.sample_file,
            force_embed=args.force_embed,
            no_quantize=args.no_quantize,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
