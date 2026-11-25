import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from engine.core import config
from engine.preprocessing.text_normalizer import normalize_sku_text, prepare_ner_text
from engine.preprocessing.weight_preprocessor import (
    calculate_dataframe_hash,
    calculate_row_hash,
    is_missing_scalar,
    populate_basic_nlp_features,
)

logger = logging.getLogger("matchops.cache")


class CacheManager:
    """Manages local metadata caching and incremental updates for the vector database."""

    def __init__(self, ner_engine, embed_engine):
        self.ner = ner_engine
        self.embedder = embed_engine
        from engine.vector_store.vector_store import VectorStore

        self.vector_store = VectorStore()

    def manage_catalog_cache(
        self,
        raw_catalog: pd.DataFrame,
        brands_df: pd.DataFrame,
        domain: str = config.DOMAIN_MARKET,
        check_for_updates: bool = False,
        force_sync: bool = False,
    ) -> tuple[pd.DataFrame, Any]:
        """Performs incremental updates to the vector store based on row-level changes."""
        cache_dir = Path(config.CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)

        hash_file = cache_dir / f"{domain}_row_hashes.json"
        processed_df_path = cache_dir / f"{domain}_catalog_metadata.pkl"

        old_hashes = {}
        if hash_file.exists():
            try:
                with hash_file.open(encoding="utf-8") as f:
                    old_hashes = json.load(f)
            except Exception:
                pass

        processed_catalog = raw_catalog.copy()

        name_counts: dict[str, int] = {}
        db_uids = []
        names = (
            raw_catalog["Name"].tolist()
            if "Name" in raw_catalog.columns
            else [""] * len(raw_catalog)
        )
        for name in names:
            raw_name = str(name).strip().lower()
            name_counts[raw_name] = name_counts.get(raw_name, 0) + 1
            uid = f"{raw_name}#occ_{name_counts[raw_name]}"
            db_uids.append(uid)
        processed_catalog["db_uid"] = db_uids

        computed_cols = ["clean_text", "weight_val", "entities", "token_count", "clean_no_weights"]
        for col in computed_cols:
            processed_catalog[col] = None

        if not processed_df_path.exists():
            if not check_for_updates and not force_sync:
                raise RuntimeError(
                    f"[CATALOG SYNC ERROR] Catalog metadata cache for domain '{domain.upper()}' was not found at '{processed_df_path}'. "
                    f"Please run 'python -m scripts.sync --cache sync' to rebuild disk caches (or 'python -m scripts.sync' for a full sync), then start the server."
                )
        else:
            try:
                cached_df = joblib.load(processed_df_path)
                if "db_uid" in cached_df.columns:
                    cached_indexed = cached_df.drop_duplicates(subset=["db_uid"]).set_index(
                        "db_uid"
                    )
                    if "entities" in cached_indexed.columns:
                        processed_catalog["entities"] = processed_catalog["db_uid"].map(
                            cached_indexed["entities"]
                        )
                    for col in ["clean_text", "clean_no_weights", "weight_val", "token_count"]:
                        if col in cached_indexed.columns:
                            processed_catalog[col] = processed_catalog["db_uid"].map(
                                cached_indexed[col]
                            )
                elif "entities" in cached_df.columns and len(cached_df) == len(processed_catalog):
                    processed_catalog["entities"] = cached_df["entities"].values
                    for col in ["clean_text", "clean_no_weights", "weight_val", "token_count"]:
                        if col in cached_df.columns:
                            processed_catalog[col] = cached_df[col].values
            except Exception as e:
                logger.warning(f"[CACHE] [{domain.upper()}] Could not load cached entities: {e}")

        if "entities" not in processed_catalog.columns:
            processed_catalog["entities"] = None

        collection_name = self.vector_store.resolve_catalog_collection_name(domain)
        collection_missing_or_empty = False
        cnt = 0
        try:
            if not self.vector_store.check_collection_exists(collection_name):
                collection_missing_or_empty = True
            else:
                cnt = self.vector_store.count_collection_points(collection_name)
                if (cnt == 0 or cnt < len(raw_catalog)) and len(raw_catalog) > 0:
                    collection_missing_or_empty = True
        except Exception as e:
            logger.warning(
                f"[CACHE] [{domain.upper()}] Could not verify Qdrant collection '{collection_name}': {e}"
            )
            collection_missing_or_empty = True

        if collection_missing_or_empty:
            logger.info(
                f"[CACHE] [{domain.upper()}] Qdrant collection '{collection_name}' has {cnt} items (expected {len(raw_catalog)}). Vectorizing and syncing catalog..."
            )
            force_sync = True
        else:
            logger.info(
                f"[CACHE] [{domain.upper()}] Checking for catalog updates (Qdrant points: {cnt}/{len(raw_catalog)})..."
            )

        if (
            not force_sync
            and not collection_missing_or_empty
            and not check_for_updates
            and "clean_text" in processed_catalog.columns
            and not processed_catalog["clean_text"].isna().any()
            and "entities" in processed_catalog.columns
            and not processed_catalog["entities"].isna().any()
            and "clean_no_weights" in processed_catalog.columns
            and not processed_catalog["clean_no_weights"].isna().any()
        ):
            logger.info(
                f"[CACHE] [{domain.upper()}] Loaded metadata from local cache and verified Qdrant collection '{collection_name}' ({cnt} points)."
            )
            return processed_catalog, self.vector_store

        name_counts = {}
        changed_indices = []
        new_hashes = {}
        row_hashes: list[str] = []
        clean_col = processed_catalog["clean_text"].tolist()
        ent_col = processed_catalog["entities"].tolist()
        records = raw_catalog.to_dict(orient="records")
        for idx, row in enumerate(records):
            raw_name = str(row.get("Name", "")).strip().lower()
            name_counts[raw_name] = name_counts.get(raw_name, 0) + 1
            uid = f"{raw_name}#occ_{name_counts[raw_name]}"

            row_hash = calculate_row_hash(row, domain=domain)
            row_hashes.append(row_hash)
            new_hashes[uid] = row_hash
            missing_meta = is_missing_scalar(clean_col[idx]) or is_missing_scalar(ent_col[idx])
            if (
                force_sync
                or collection_missing_or_empty
                or uid not in old_hashes
                or old_hashes[uid] != row_hash
                or missing_meta
            ):
                changed_indices.append(idx)

        if not changed_indices:
            logger.info(f"[CACHE] [{domain.upper()}] No changes detected. Database is up to date.")
            return processed_catalog, self.vector_store

        logger.info(
            f"[CACHE] [{domain.upper()}] Detected {len(changed_indices)} changed/new/unindexed rows. Syncing..."
        )
        changed_df = processed_catalog.loc[changed_indices].copy()
        changed_df["row_hash"] = [row_hashes[i] for i in changed_indices]

        missing_clean_changed = (
            changed_df["clean_text"].isna()
            | (changed_df["clean_text"].astype(str) == "None")
            | (changed_df["clean_text"].astype(str).str.strip() == "")
            | changed_df["clean_no_weights"].isna()
            | (changed_df["clean_no_weights"].astype(str) == "None")
            | changed_df["token_count"].isna()
        )
        if missing_clean_changed.any():
            logger.info("[CACHE]      - Normalizing Text, Weights & Auxiliary Features...")
            sub_features = populate_basic_nlp_features(changed_df.loc[missing_clean_changed].copy())
            for col in ("clean_text", "weight_val", "token_count", "clean_no_weights"):
                changed_df.loc[missing_clean_changed, col] = sub_features[col]

        missing_ent_mask = (
            changed_df["entities"].isna()
            if "entities" in changed_df.columns
            else pd.Series(True, index=changed_df.index)
        )
        if missing_ent_mask.any():
            logger.info("[CACHE]      - Extracting Entities...")
            missing_df = changed_df.loc[missing_ent_mask]
            target_col = "Brand" if domain == config.DOMAIN_MARKET else "Flavor"
            target_label = self.ner.target_label
            entities_list: list[dict | None] = []

            if target_col in missing_df.columns:
                for raw_val in missing_df[target_col].tolist():
                    val = str(raw_val or "").strip()
                    if val and val.lower() != "nan":
                        val_set = {v.strip().lower() for v in val.split(",") if v.strip()}
                        entities_list.append({target_label: val_set})
                    else:
                        entities_list.append(None)

            miss_ner_positions = (
                [i for i, e in enumerate(entities_list) if e is None]
                if entities_list
                else list(range(len(missing_df)))
            )
            if not entities_list:
                entities_list = [None] * len(missing_df)

            if miss_ner_positions:
                raw_names = missing_df["Name"].tolist()
                texts_for_ner = [prepare_ner_text(str(raw_names[i])) for i in miss_ner_positions]
                ner_results = self.ner.batch_extract_entities(
                    texts_for_ner, batch_size=config.ONNX_BATCH_SIZE
                )
                for pos, ner_ent in zip(miss_ner_positions, ner_results):
                    entities_list[pos] = ner_ent

            changed_df.loc[missing_ent_mask, "entities"] = pd.Series(
                entities_list, index=missing_df.index, dtype=object
            )

        for col in computed_cols:
            processed_catalog.loc[changed_indices, col] = changed_df[col]

        has_sparse = bool(self.embedder and getattr(self.embedder, "has_learned_sparse", False))
        vec_desc = "Dense + BM25 + Sparse" if has_sparse else "Dense + BM25"
        logger.info(
            f"[CACHE]      - Vectorizing Data ({len(changed_df)} items with {config.EMBEDDING_MODEL} {vec_desc})..."
        )
        catalog_names = changed_df["clean_text"].tolist()
        catalog_descs = (
            [
                normalize_sku_text(str(d))
                if str(d).strip().lower() not in ("nan", "none", "<na>")
                else ""
                for d in changed_df["Description"].tolist()
            ]
            if "Description" in changed_df.columns
            else [""] * len(changed_df)
        )
        cat_col = config.resolve_input_category_col(changed_df.columns)
        catalog_cats = (
            [
                normalize_sku_text(str(c))
                if str(c).strip().lower() not in ("nan", "none", "<na>")
                else ""
                for c in changed_df[cat_col].tolist()
            ]
            if cat_col and cat_col in changed_df.columns
            else [""] * len(changed_df)
        )
        encoded = self.embedder.embed_weighted_sku(
            catalog_names, catalog_descs, catalog_cats, weights=config.MATCHER_WEIGHTS
        )
        dense_embeddings = encoded["dense"]
        bm25_weights = encoded.get("bm25")
        sparse_weights = encoded.get("sparse")

        logger.info(
            f"[CACHE]      - Upserting {len(changed_df)} items to Qdrant collection '{collection_name}'..."
        )
        self.vector_store.upsert_catalog_points(
            changed_df,
            dense_embeddings,
            sparse_weights=sparse_weights,
            bm25_weights=bm25_weights,
            domain=domain,
        )
        logger.info(
            f"[CACHE] [{domain.upper()}] ✓ Successfully synced {len(changed_df)} catalog items to Qdrant."
        )

        logger.info(
            f"[CACHE] [{domain.upper()}] Saving processed catalog metadata ({len(processed_catalog)} rows) to '{processed_df_path}'."
        )
        joblib.dump(processed_catalog, processed_df_path)

        with hash_file.open("w", encoding="utf-8") as f:
            json.dump(new_hashes, f)

        logger.info(f"[CACHE] [{domain.upper()}] Sync complete.")
        return processed_catalog, self.vector_store

    def get_or_build_bt_gk_cache(
        self, cat_df: pd.DataFrame, domain: str, build_fn: Callable
    ) -> dict:
        """
        Returns cached classifier data (BT-GK maps) or builds them if the catalog has changed.
        """
        cache_dir = Path(config.CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{domain}_bt_gk_cache.pkl"
        hash_path = cache_dir / f"{domain}_bt_gk_hash.txt"

        current_hash = calculate_dataframe_hash(cat_df, domain=domain)

        if cache_path.exists() and hash_path.exists():
            try:
                stored_hash = hash_path.read_text(encoding="utf-8").strip()
                if stored_hash == current_hash:
                    cached = joblib.load(cache_path)
                    logger.info(
                        f"[CACHE] [{domain.upper()}] ✓ Classifier BT-GK map loaded from cache."
                    )
                    return cached
            except Exception:
                pass

        logger.info(f"[CACHE] [{domain.upper()}] Building BT-GK map from catalog...")
        build_result = build_fn(cat_df)

        if len(build_result) == 3:
            bt_gk_map, umbrella, third_tag_map = build_result
            payload = {"bt_gk_map": bt_gk_map, "umbrella": umbrella, "third_tag_map": third_tag_map}
        else:
            bt_gk_map, umbrella = build_result
            payload = {"bt_gk_map": bt_gk_map, "umbrella": umbrella}

        joblib.dump(payload, cache_path)
        hash_path.write_text(current_hash or "", encoding="utf-8")

        return payload
