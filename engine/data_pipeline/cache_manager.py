import hashlib
import json
import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple

import joblib
import pandas as pd

from engine import config

logger = logging.getLogger("matchops.cache")

def clean_price(val) -> Optional[float]:
    """Safely cleans and converts a price value to a float."""
    if pd.isna(val) or val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val_str = val.strip().replace("$", "").replace(",", "")
        if not val_str:
            return None
        try:
            return float(val_str)
        except ValueError:
            return None
    return None

def calculate_row_hash(row, domain: str = config.DOMAIN_MARKET) -> str:
    """Generates a stable MD5 hash for a single row."""
    row_dict = row if isinstance(row, dict) else row.to_dict()
    
    # Standard list of columns we care about for semantic hashing
    cols = ["Name", "basictype", "Generic keywords", "Price", "Description"]
    if domain == config.DOMAIN_FOOD:
        cols.extend(["region", "Flavor"])
    else:
        cols.extend(["category", "Brand"])
        
    cleaned = {}
    for c in cols:
        val = row_dict.get(c, "")
        if pd.isna(val) or val is None:
            val = ""
        elif c == "Price":
            p = clean_price(val)
            val = f"{p:.2f}" if p is not None else ""
        elif isinstance(val, (int, float)):
            val = f"{float(val):.2f}"
        else:
            val = str(val).strip()
        cleaned[c] = val
        
    # JSON dump with sorted keys guarantees deterministic serialization order
    row_str = json.dumps(cleaned, sort_keys=True) + config.CACHE_SALT
    return hashlib.md5(row_str.encode("utf-8")).hexdigest()

def calculate_df_hash(df: pd.DataFrame, domain: str = config.DOMAIN_MARKET) -> Optional[str]:
    """Stable, cross-session deterministic hash of a DataFrame's content, ignoring order and formatting quirks."""
    if df is None or df.empty:
        return None
    
    # Select only relevant columns
    cols = ["Name", "basictype", "Generic keywords", "Price", "Description"]
    if domain == config.DOMAIN_FOOD:
        cols.extend(["region", "Flavor"])
    else:
        cols.extend(["category", "Brand"])
        
    existing_cols = [c for c in cols if c in df.columns]
    temp_df = df[existing_cols].copy()
    
    # Drop rows where Name is missing/empty
    temp_df = temp_df[temp_df["Name"].astype(str).str.strip() != ""]
    temp_df = temp_df[temp_df["Name"].notna()]
    
    # Standardize text columns (strip, lowercase) and floats (format to 2 decimals)
    for col in temp_df.columns:
        if col == "Price":
            temp_df[col] = pd.to_numeric(temp_df[col], errors="coerce").fillna(0.0).round(2)
        else:
            temp_df[col] = temp_df[col].astype(str).str.strip().str.lower()
            
    # Sort by Name to make it independent of row ordering in Google Sheets
    temp_df = temp_df.sort_values(by="Name").reset_index(drop=True)
    
    # Stable JSON list format
    df_json = temp_df.to_json(orient="records")
    return hashlib.md5(df_json.encode("utf-8")).hexdigest()

class CacheManager:
    """Manages local metadata caching and incremental updates for the vector database."""

    def __init__(self, ner_engine, embed_engine):
        self.ner = ner_engine
        self.embedder = embed_engine
        from engine.data_pipeline.vector_store import VectorStore
        self.vector_store = VectorStore()

    def _hash_row(self, row, domain: str = config.DOMAIN_MARKET) -> str:
        """Generates a stable MD5 hash for a single row."""
        return calculate_row_hash(row, domain=domain)

    @staticmethod
    def _hash_df(df: pd.DataFrame, domain: str = config.DOMAIN_MARKET) -> Optional[str]:
        """Stable, cross-session deterministic hash of a DataFrame's content, ignoring order and formatting quirks."""
        return calculate_df_hash(df, domain=domain)

    def manage_catalog_cache(
        self, raw_catalog: pd.DataFrame, brands_df: pd.DataFrame, domain: str = config.DOMAIN_MARKET, check_for_updates: bool = False, force_sync: bool = False
    ) -> Tuple[pd.DataFrame, Any]:
        """Performs incremental updates to the vector store based on row-level changes."""
        from engine.nlp.text_cleaner import TextPipeline
        os.makedirs(config.CACHE_DIR, exist_ok=True)

        hash_file = os.path.join(config.CACHE_DIR, f"{domain}_row_hashes.json")
        processed_df_path = os.path.join(config.CACHE_DIR, f"{domain}_catalog_metadata.pkl")

        old_hashes = {}
        if os.path.exists(hash_file):
            try:
                with open(hash_file, "r") as f:
                    old_hashes = json.load(f)
            except Exception:
                pass

        processed_catalog = raw_catalog.copy()
        
        name_counts = {}
        db_uids = []
        for idx, row in raw_catalog.iterrows():
            raw_name = str(row.get("Name", "")).strip().lower()
            name_counts[raw_name] = name_counts.get(raw_name, 0) + 1
            uid = f"{raw_name}#occ_{name_counts[raw_name]}"
            db_uids.append(uid)
        processed_catalog["db_uid"] = db_uids

        computed_cols = ["clean_text", "weight_val", "entities", "token_count", "clean_no_weights"]

        # Normalize text and clean units upfront (instant vectorized operation, ~0.05s)
        expected_clean = processed_catalog["Name"].fillna("").astype(str).apply(
            lambda x: TextPipeline.normalize_final(TextPipeline.standardize_units(x))
        )
        processed_catalog["clean_text"] = expected_clean
        processed_catalog["clean_no_weights"] = processed_catalog["clean_text"].apply(TextPipeline.strip_weights)
        processed_catalog["weight_val"] = processed_catalog["clean_text"].apply(TextPipeline.extract_weight_feature)
        processed_catalog["token_count"] = processed_catalog["clean_text"].fillna("").astype(str).apply(lambda s: len(s.split()))

        # Attempt to load pre-calculated entities and metadata from disk cache
        if not os.path.exists(processed_df_path):
            if not check_for_updates and not force_sync:
                raise RuntimeError(
                    f"[CATALOG SYNC ERROR] Catalog metadata cache for domain '{domain.upper()}' was not found at '{processed_df_path}'. "
                    f"Please run 'python -m backend.sync_catalog --cache' to rebuild disk caches (or 'python -m backend.sync_catalog' for a full sync), then start the server."
                )
        else:
            try:
                cached_df = joblib.load(processed_df_path)
                if "db_uid" in cached_df.columns:
                    cached_indexed = cached_df.drop_duplicates(subset=["db_uid"]).set_index("db_uid")
                    if "entities" in cached_indexed.columns:
                        processed_catalog["entities"] = processed_catalog["db_uid"].map(cached_indexed["entities"])
                elif "entities" in cached_df.columns and len(cached_df) == len(processed_catalog):
                    processed_catalog["entities"] = cached_df["entities"].values
            except Exception as e:
                logger.warning(f"[CACHE] [{domain.upper()}] Could not load cached entities: {e}")

        if "entities" not in processed_catalog.columns:
            processed_catalog["entities"] = None

        # Verify Qdrant collection status
        collection_name = self.vector_store._get_collection_name(domain)
        collection_missing_or_empty = False
        try:
            if not self.vector_store.client.collection_exists(collection_name):
                collection_missing_or_empty = True
            else:
                cnt = self.vector_store.client.count(collection_name).count
                if cnt == 0 and len(raw_catalog) > 0:
                    collection_missing_or_empty = True
        except Exception as e:
            logger.warning(f"[CACHE] [{domain.upper()}] Could not verify Qdrant collection '{collection_name}': {e}")
            collection_missing_or_empty = True

        if collection_missing_or_empty:
            logger.info(f"[CACHE] [{domain.upper()}] Qdrant collection '{collection_name}' is missing or empty. Vectorizing and syncing catalog...")
            force_sync = True
        else:
            logger.info(f"[CACHE] [{domain.upper()}] Checking for catalog updates...")

        # If we do not need to check for updates, collection exists in Qdrant, and cache is populated, return immediately
        if not force_sync and not collection_missing_or_empty and not check_for_updates and "clean_text" in processed_catalog.columns and not processed_catalog["clean_text"].isna().all():
            logger.info(f"[CACHE] [{domain.upper()}] Loaded metadata from local cache and verified Qdrant collection '{collection_name}'.")
            return processed_catalog, self.vector_store

        # Identify changed or new rows using stable name-based UIDs
        name_counts = {}
        changed_indices = []
        new_hashes = {}
        records = raw_catalog.to_dict(orient="records")
        for idx, row in enumerate(records):
            raw_name = str(row.get("Name", "")).strip().lower()
            name_counts[raw_name] = name_counts.get(raw_name, 0) + 1
            # Unique stable identity for duplicate names
            uid = f"{raw_name}#occ_{name_counts[raw_name]}"
            
            row_hash = self._hash_row(row, domain=domain)
            new_hashes[uid] = row_hash
            if force_sync or collection_missing_or_empty or uid not in old_hashes or old_hashes[uid] != row_hash:
                changed_indices.append(idx)

        if not changed_indices:
            logger.info(f"[CACHE] [{domain.upper()}] No changes detected. Database is up to date.")
            return processed_catalog, self.vector_store

        logger.info(f"[CACHE] [{domain.upper()}] Detected {len(changed_indices)} changed/new/unindexed rows. Syncing...")
        changed_df = processed_catalog.loc[changed_indices].copy()

        # 1. Text Normalization (only compute for rows missing clean_text)
        missing_clean_changed = (
            changed_df["clean_text"].isna() | 
            (changed_df["clean_text"].astype(str) == "None") | 
            (changed_df["clean_text"].astype(str).str.strip() == "")
        )
        if missing_clean_changed.any():
            logger.info("[CACHE]      - Normalizing Text & Weights...")
            changed_df.loc[missing_clean_changed, "clean_text"] = changed_df.loc[missing_clean_changed, "Name"].astype(str).apply(
                lambda x: TextPipeline.normalize_final(TextPipeline.standardize_units(x))
            )
            changed_df.loc[missing_clean_changed, "weight_val"] = changed_df.loc[missing_clean_changed, "clean_text"].apply(TextPipeline.extract_weight_feature)

        # 2. Entity Extraction (only compute for rows missing entities)
        needs_entities = False
        if "entities" not in changed_df.columns:
            needs_entities = True
        else:
            sample_val = changed_df["entities"].dropna()
            if len(sample_val) == 0:
                needs_entities = True

        if needs_entities:
            logger.info("[CACHE]      - Extracting Entities...")
            target_col = "Brand" if domain == config.DOMAIN_MARKET else "Flavor"
            target_label = self.ner.target_label
            entities_list = []

            if target_col in changed_df.columns:
                # Fast path: read flavors/brands directly from the pre-filled sheet column.
                # Rows missing the value fall through to the NER engine (Layer 1 dict + Layer 2 NER).
                for _, row in changed_df.iterrows():
                    val = str(row.get(target_col, "")).strip()
                    if val and val.lower() != "nan":
                        val_set = {v.strip().lower() for v in val.split(",") if v.strip()}
                        entities_list.append({target_label: val_set})
                    else:
                        entities_list.append(None)

                if None in entities_list:
                    texts_for_ner = [TextPipeline.prep_for_ner(str(row["Name"])) for _, row in changed_df.iterrows()]
                    ner_results = self.ner.batch_extract_entities(texts_for_ner, batch_size=config.EMBED_BATCH_SIZE)
                    for i in range(len(entities_list)):
                        if entities_list[i] is None:
                            entities_list[i] = ner_results[i]
            else:
                texts_for_ner = [TextPipeline.prep_for_ner(str(t)) for t in changed_df["Name"].tolist()]
                entities_list = self.ner.batch_extract_entities(texts_for_ner, batch_size=config.EMBED_BATCH_SIZE)
                
            changed_df["entities"] = entities_list

        # 3. Pre-calculate Auxiliary Features
        missing_aux = (
            "clean_no_weights" not in changed_df.columns or
            changed_df["clean_no_weights"].isna().any() |
            (changed_df["clean_no_weights"].astype(str) == "None").any()
        )
        if missing_aux:
            logger.info("[CACHE]      - Pre-calculating Auxiliary Features...")
            changed_df["token_count"] = changed_df["clean_text"].fillna("").astype(str).apply(lambda s: len(s.split()))
            changed_df["clean_no_weights"] = changed_df["clean_text"].apply(TextPipeline.strip_weights)

        # Merge pre-calculated columns back into the main catalog
        for col in computed_cols:
            processed_catalog.loc[changed_indices, col] = changed_df[col]

        # 4. Hybrid Vectorization
        logger.info(f"[CACHE]      - Vectorizing Data ({len(changed_df)} items with BGE-M3 Dense + Sparse)...")
        catalog_names = changed_df["clean_text"].tolist()
        catalog_descs = changed_df["Description"].tolist() if "Description" in changed_df.columns else [""] * len(changed_df)
        cat_col = config.get_third_tag_col(domain)
        catalog_cats = changed_df[cat_col].tolist() if cat_col in changed_df.columns else [""] * len(changed_df)
        encoded = self.embedder.embed_weighted_sku(
            catalog_names, catalog_descs, catalog_cats,
            weights=config.MATCHER_WEIGHTS
        )
        dense_embeddings = encoded["dense"]
        sparse_weights = encoded["sparse"]

        # 5. Persistent Storage Sync
        logger.info(f"[CACHE]      - Upserting {len(changed_df)} items to Qdrant collection '{collection_name}'...")
        self.vector_store.sync(changed_df, dense_embeddings, sparse_weights, domain=domain)
        logger.info(f"[CACHE] [{domain.upper()}] ✓ Successfully synced {len(changed_df)} catalog items to Qdrant.")

        # 6. Finalize Local Cache (update SQLite DB)
        import sqlite3
        conn = sqlite3.connect(config.DB_PATH)
        try:
            for list_idx, (orig_idx, row) in enumerate(changed_df.iterrows()):
                entities_val = row.get("entities")
                ent_json = None
                if entities_val:
                    # Convert sets to lists inside dictionary for JSON serialization
                    ent_json = json.dumps([ {k: list(v) if isinstance(v, set) else v for k, v in item.items()} if item else None for item in entities_val ] if isinstance(entities_val, list) else {k: list(v) if isinstance(v, set) else v for k, v in entities_val.items()})

                w_val = row.get("weight_val")
                # Since weight_val is a tuple (value, unit, type), serialize it as JSON for SQLite storage
                w_json = json.dumps(w_val) if isinstance(w_val, (tuple, list)) else w_val

                conn.execute("""
                    UPDATE catalog_items
                    SET clean_text = ?,
                        weight_val = ?,
                        entities_json = ?,
                        token_count = ?,
                        clean_no_weights = ?
                    WHERE domain = ? AND row_hash = ?
                """, (
                    row.get("clean_text"),
                    w_json,
                    ent_json,
                    int(row.get("token_count")) if not pd.isna(row.get("token_count")) else None,
                    row.get("clean_no_weights"),
                    domain,
                    row.get("row_hash")
                ))
            conn.commit()
            logger.info(f"[CACHE] [{domain.upper()}] Saved computed metadata for {len(changed_df)} rows directly to SQLite.")
        except Exception as db_err:
            conn.rollback()
            logger.error(f"[CACHE] [{domain.upper()}] Failed to commit NLP metadata to SQLite: {db_err}")
        finally:
            conn.close()

        with open(hash_file, "w") as f:
            json.dump(new_hashes, f)

        logger.info(f"[CACHE] [{domain.upper()}] Sync complete.")
        return processed_catalog, self.vector_store

    def get_or_build_bt_gk_cache(self, cat_df: pd.DataFrame, domain: str, build_fn: Callable) -> dict:
        """
        Returns cached classifier data (BT-GK maps) or builds them if the catalog has changed.
        """
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(config.CACHE_DIR, f"{domain}_bt_gk_cache.pkl")
        hash_path = os.path.join(config.CACHE_DIR, f"{domain}_bt_gk_hash.txt")

        current_hash = self._hash_df(cat_df, domain=domain)

        # Validate cache by comparing catalog hashes
        if os.path.exists(cache_path) and os.path.exists(hash_path):
            try:
                stored_hash = open(hash_path).read().strip()
                if stored_hash == current_hash:
                    cached = joblib.load(cache_path)
                    logger.info(f"[CACHE] [{domain.upper()}] ✓ Classifier BT-GK map loaded from cache.")
                    return cached
            except Exception:
                pass

        # Rebuild on cache miss
        logger.info(f"[CACHE] [{domain.upper()}] Building BT-GK map from catalog...")
        build_result = build_fn(cat_df)
        
        if len(build_result) == 3:
            bt_gk_map, umbrella, third_tag_map = build_result
            payload = {"bt_gk_map": bt_gk_map, "umbrella": umbrella, "third_tag_map": third_tag_map}
        else:
            bt_gk_map, umbrella = build_result
            payload = {"bt_gk_map": bt_gk_map, "umbrella": umbrella}

        joblib.dump(payload, cache_path)
        with open(hash_path, "w") as f:
            f.write(current_hash or "")

        return payload
