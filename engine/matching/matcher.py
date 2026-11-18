import logging
import re
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from engine import config
from engine.nlp.text_cleaner import TextPipeline
from engine.utils.flavor_utils import build_food_flavors_info, triage_input_flavors
from engine.utils.weight_utils import resolve_weight_bypass_candidate

logger = logging.getLogger("matchops.matcher")


class SKUMatcher:
    """Main SKU matching pipeline orchestrator."""

    def __init__(
        self,
        catalog_df: pd.DataFrame,
        brands_df: pd.DataFrame,
        ner_engine,
        embed_engine,
        cache_manager,
        logic_gates,
        domain: str = config.DOMAIN_MARKET,
        classifier=None,
        check_for_updates: bool = False,
        force_sync: bool = False,
    ):
        self.ner = ner_engine
        self.embedder = embed_engine
        self.rules = logic_gates
        self.domain = domain
        self.classifier = classifier
        self.brands_df = brands_df

        # Build flavor categories dictionary and generic term lists for food domain
        self.flavor_categories = {}
        self.mixed_terms = {"mixed", "mix"}
        self.seafood_terms = {"seafood", "sea food"}
        self.veg_terms = {"vegetable", "vegetables", "veg", "veggie", "vege", "vegetarian"}

        if self.domain == config.DOMAIN_FOOD and brands_df is not None and not brands_df.empty:
            _, _, _, _, self.flavor_categories = build_food_flavors_info(brands_df)

            # Load special category terms dynamically from the loaded sheet data
            def get_all_terms_for(name: str) -> set:
                terms = {name.lower()}
                row = brands_df[brands_df["Flavor Name"].str.lower() == name.lower()]
                if not row.empty:
                    aliases_str = str(row.iloc[0].get("Aliases", ""))
                    if (
                        aliases_str
                        and aliases_str.lower() != "none"
                        and aliases_str.lower() != "nan"
                    ):
                        aliases = [x.strip().lower() for x in aliases_str.split(",") if x.strip()]
                        terms.update(aliases)
                # Apply pluralization variations to ensure coverage
                pluralized = []
                for t in terms:
                    if t.endswith("y"):
                        pluralized.append(t[:-1] + "ies")
                    elif t.endswith("s"):
                        pass
                    else:
                        pluralized.append(t + "s")
                terms.update(pluralized)
                return terms

            self.mixed_terms = get_all_terms_for("Mixed")
            self.seafood_terms = get_all_terms_for("Seafood")
            self.veg_terms = get_all_terms_for("Vegetable")

        def _compile_terms(terms_set):
            if not terms_set:
                return None
            sorted_t = sorted(
                [t.lower().strip() for t in terms_set if t.strip()],
                key=lambda x: (len(x), x),
                reverse=True,
            )
            if not sorted_t:
                return None
            return re.compile(
                r"\b(" + "|".join(re.escape(t) for t in sorted_t) + r")\b", re.IGNORECASE
            )

        self._mixed_pattern = _compile_terms(self.mixed_terms)
        self._seafood_pattern = _compile_terms(self.seafood_terms)
        self._veg_pattern = _compile_terms(self.veg_terms)

        if self.flavor_categories:
            specific_terms = [
                term
                for term in self.flavor_categories.keys()
                if term not in self.mixed_terms
                and term not in self.seafood_terms
                and term not in self.veg_terms
            ]
            self._specific_flavor_pattern = _compile_terms(specific_terms)
        else:
            self._specific_flavor_pattern = None

        # Sync catalog with cache and vector store
        self.raw_catalog, self.vector_store = cache_manager.manage_catalog_cache(
            catalog_df,
            brands_df,
            domain=domain,
            check_for_updates=check_for_updates,
            force_sync=force_sync,
        )
        self.raw_catalog = self.raw_catalog.reset_index(drop=True)
        self._catalog_records: list[dict[str, Any]] = self.raw_catalog.to_dict(orient="records")
        self._catalog_bts_lower: list[str] = [
            str(rec.get("basictype", rec.get("BasicType", "")) or "").strip().lower()
            for rec in self._catalog_records
        ]

        # Build lookup maps for fast O(1) exact matching and O(1) token-sort fuzzy bypass
        self.exact_match_map: dict[str, int] = {}
        self.token_sorted_map: dict[str, list[int]] = {}

        # Perf: iterate pre-extracted column lists instead of DataFrame.iterrows(), which
        # allocates a pandas Series per row. Benchmarked on the cached food catalog
        # (79k rows): 1.44s -> 0.14s (~10x); market (9k rows): 0.15s -> 0.009s.
        # raw_catalog has a RangeIndex (reset above), so enumerate() yields the same positional i.
        n_rows = len(self.raw_catalog)
        cols = self.raw_catalog.columns
        clean_col = (
            self.raw_catalog["clean_text"].tolist() if "clean_text" in cols else [None] * n_rows
        )
        no_w_col = (
            self.raw_catalog["clean_no_weights"].tolist()
            if "clean_no_weights" in cols
            else [None] * n_rows
        )
        name_col = self.raw_catalog["Name"].tolist() if "Name" in cols else [""] * n_rows

        for i, (raw_clean, raw_no_w, raw_name) in enumerate(zip(clean_col, no_w_col, name_col)):
            clean_txt = str(raw_clean or "").strip()
            if not clean_txt or clean_txt == "None":
                clean_txt = TextPipeline.normalize_final(
                    TextPipeline.standardize_units(str(raw_name))
                )
            no_weights = str(raw_no_w or TextPipeline.strip_weights(clean_txt)).strip()

            if clean_txt and clean_txt != "None":
                if clean_txt not in self.exact_match_map:
                    self.exact_match_map[clean_txt] = []
                self.exact_match_map[clean_txt].append(i)

            sorted_tokens = " ".join(sorted(no_weights.split())).strip()
            if sorted_tokens and sorted_tokens != "None":
                if sorted_tokens not in self.token_sorted_map:
                    self.token_sorted_map[sorted_tokens] = []
                self.token_sorted_map[sorted_tokens].append(i)

    def search_candidates(
        self,
        input_vec_dense: np.ndarray,
        input_vec_sparse: dict[str, float] | None = None,
        bt_filter: str | None = None,
        input_vec_bm25: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Retrieves top-K candidates from the vector store using hybrid RRF search, optionally filtered by BT."""
        hits = self.vector_store.search_from_vectors(
            dense_vec=input_vec_dense,
            sparse_vec_dict=input_vec_sparse,
            domain=self.domain,
            top_k=config.TOP_K_RETRIEVAL,
            bt_filter=bt_filter,
            bm25_vec_dict=input_vec_bm25,
        )
        return pd.DataFrame(hits) if hits else pd.DataFrame()

    def process_inputs(
        self,
        input_df: pd.DataFrame,
        progress_callback: Callable | None = None,
        chunk_size: int | None = None,
    ) -> pd.DataFrame:
        """Matches input SKUs against the catalog with automatic chunking to keep RAM flat and prevent OOM."""
        total_skus = len(input_df)
        if total_skus == 0:
            return pd.DataFrame()

        effective_chunk_size = chunk_size or getattr(config, "MATCH_CHUNK_SIZE", 250)

        if total_skus <= effective_chunk_size:
            return self._process_single_chunk(input_df, progress_callback=progress_callback)

        num_chunks = (total_skus + effective_chunk_size - 1) // effective_chunk_size
        logger.info(
            f"[MATCH] Processing {total_skus} SKUs in {num_chunks} chunks (chunk_size={effective_chunk_size})..."
        )

        results = []
        for chunk_idx, chunk_start in enumerate(range(0, total_skus, effective_chunk_size)):
            chunk_end = min(chunk_start + effective_chunk_size, total_skus)
            chunk_df = input_df.iloc[chunk_start:chunk_end]

            def chunk_cb(
                pct,
                msg=None,
                _start=chunk_start,
                _len=len(chunk_df),
                _idx=chunk_idx,
                _end=chunk_end,
            ):
                if progress_callback:
                    overall_pct = ((_start + (float(pct) / 100.0) * _len) / total_skus) * 100.0
                    progress_callback(
                        overall_pct,
                        msg or f"Chunk {_idx + 1}/{num_chunks} ({_end}/{total_skus} SKUs)",
                    )

            chunk_res = self._process_single_chunk(chunk_df, progress_callback=chunk_cb)
            results.append(chunk_res)

        if progress_callback:
            progress_callback(100.0, "Matching completed.")

        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    def _process_single_chunk(
        self, input_df: pd.DataFrame, progress_callback: Callable | None = None
    ) -> pd.DataFrame:
        """Matches a single chunk of input SKUs against the catalog."""
        results = []
        logger.debug(f"[MATCH] Pre-processing {len(input_df)} input SKUs...")

        raw_names, clean_inputs, prices, input_no_weights_list, input_w_data_list = (
            [],
            [],
            [],
            [],
            [],
        )
        descriptions, categories = [], []
        cols = input_df.columns
        name_col = "Name" if "Name" in cols else (cols[0] if len(cols) > 0 else None)
        desc_col = (
            "Description"
            if "Description" in cols
            else ("description" if "description" in cols else None)
        )
        cat_col = "Category" if "Category" in cols else ("category" if "category" in cols else None)
        price_col = "Price" if "Price" in cols else ("price" if "price" in cols else None)

        n_inputs = len(input_df)
        names = input_df[name_col].tolist() if name_col else [""] * n_inputs
        descs = input_df[desc_col].tolist() if desc_col else [""] * n_inputs
        cats = input_df[cat_col].tolist() if cat_col else [""] * n_inputs
        raw_prices = input_df[price_col].tolist() if price_col else [0] * n_inputs

        for raw_name, raw_desc, raw_cat, raw_price in zip(names, descs, cats, raw_prices):
            name = str(raw_name if (raw_name is not None and not pd.isna(raw_name)) else "")
            raw_names.append(name)
            desc = str(raw_desc if (raw_desc is not None and not pd.isna(raw_desc)) else "")
            cat = str(raw_cat if (raw_cat is not None and not pd.isna(raw_cat)) else "")
            descriptions.append(desc)
            categories.append(cat)

            clean_input = TextPipeline.normalize_final(TextPipeline.standardize_units(name))
            clean_inputs.append(clean_input)
            input_no_weights_list.append(TextPipeline.strip_weights(clean_input))
            input_w_data_list.append(TextPipeline.extract_weight_feature(clean_input))

            # Extract price for logic gate validation
            try:
                prices.append(
                    float(re.sub(r"[^\d\.]", "", str(raw_price)))
                    if raw_price and not pd.isna(raw_price)
                    else 0.0
                )
            except (ValueError, TypeError):
                prices.append(0.0)

        cat_records = getattr(self, "_catalog_records", None)
        if cat_records is None or len(cat_records) != len(self.raw_catalog):
            cat_records = self.raw_catalog.to_dict(orient="records")
            self._catalog_records = cat_records
            self._catalog_bts_lower = [
                str(rec.get("basictype", rec.get("BasicType", "")) or "").strip().lower()
                for rec in cat_records
            ]
        cat_bts_lower = self._catalog_bts_lower

        # Step 1: O(1) Exact & Token-Sorted Bypass Pre-Check
        # Unambiguous single-BT bypasses resolve immediately without running ONNX Bi-Encoder or ArcFace BT.
        ai_indices: list[int] = []
        pending_exact: dict[int, list[int]] = {}
        pending_fuzzy: dict[int, list[int]] = {}
        needs_bt_or_ai_indices: list[int] = []
        bypass_results: list[tuple | None] = [None] * len(raw_names)
        logger.info("[MATCH] Executing exact-match and deterministic rules fast-path...")

        for i, clean_input in enumerate(clean_inputs):
            input_no_weights = input_no_weights_list[i]
            input_w_data = input_w_data_list[i]

            if not clean_input or clean_input == "None":
                ai_indices.append(i)
                needs_bt_or_ai_indices.append(i)
                continue

            # Exact text match lookup
            cat_indices = self.exact_match_map.get(clean_input)

            if cat_indices is None and clean_input.endswith(" mixed"):
                cat_indices = self.exact_match_map.get(clean_input[:-6])

            if cat_indices is not None:
                if isinstance(cat_indices, int):
                    cat_indices = [cat_indices]

                if not cat_indices:
                    ai_indices.append(i)
                    needs_bt_or_ai_indices.append(i)
                    continue

                if (
                    len(cat_indices) == 1
                    or self.classifier is None
                    or len({cat_bts_lower[idx] for idx in cat_indices}) <= 1
                ):
                    bypass_results[i] = (
                        cat_records[cat_indices[0]],
                        1.0,
                        "High Confidence",
                        "Exact Text Match",
                    )
                else:
                    pending_exact[i] = cat_indices
                    needs_bt_or_ai_indices.append(i)
                continue

            # Early fuzzy search on weight-stripped text (O(1) Hash Map Optimization)
            sorted_input_tokens = " ".join(sorted(str(input_no_weights).split()))
            cand_indices = self.token_sorted_map.get(sorted_input_tokens)

            if cand_indices:
                if isinstance(cand_indices, int):
                    cand_indices = [cand_indices]

                if (
                    len(cand_indices) == 1
                    or self.classifier is None
                    or len({cat_bts_lower[idx] for idx in cand_indices}) <= 1
                ):
                    selected_idx, weight_reason = resolve_weight_bypass_candidate(
                        cat_records, cand_indices, input_w_data
                    )
                    bypass_results[i] = (
                        cat_records[selected_idx],
                        1.0,
                        "High Confidence",
                        f"Fuzzy Match (100%){weight_reason}",
                    )
                else:
                    pending_fuzzy[i] = cand_indices
                    needs_bt_or_ai_indices.append(i)
            else:
                ai_indices.append(i)
                needs_bt_or_ai_indices.append(i)

        # Step 1B: Batch encode & predict BT only for ambiguous multi-BT bypasses and non-bypassed SKUs
        input_dense_embs: dict[int, np.ndarray] = {}
        input_bm25_embs: dict[int, dict[str, float]] = {}
        input_sparse_embs: dict[int, dict[str, float] | None] = {}
        bt_filters: list[str | None] = [None] * len(raw_names)

        if needs_bt_or_ai_indices:
            sub_clean = [clean_inputs[idx] for idx in needs_bt_or_ai_indices]
            sub_descs = [descriptions[idx] for idx in needs_bt_or_ai_indices]
            sub_cats = [categories[idx] for idx in needs_bt_or_ai_indices]
            encoded = self.embedder.embed_weighted_sku(
                sub_clean, sub_descs, sub_cats, weights=config.MATCHER_WEIGHTS
            )
            sub_dense = encoded["dense"]
            sub_bm25 = encoded.get("bm25") or []
            sub_sparse = encoded.get("sparse") or []
            for pos, orig_idx in enumerate(needs_bt_or_ai_indices):
                input_dense_embs[orig_idx] = sub_dense[pos]
                input_bm25_embs[orig_idx] = sub_bm25[pos] if pos < len(sub_bm25) else {}
                input_sparse_embs[orig_idx] = sub_sparse[pos] if pos < len(sub_sparse) else None

            if self.classifier:
                sub_prices = [prices[idx] for idx in needs_bt_or_ai_indices]
                sub_raw_names = [raw_names[idx] for idx in needs_bt_or_ai_indices]
                bt_preds = self.classifier.batch_predict_bt(
                    np.array(sub_dense),
                    sub_prices,
                    sku_names=sub_raw_names,
                    sku_descriptions=sub_descs,
                )
                for pos, p_res in enumerate(bt_preds):
                    if p_res:
                        bt_tag, conf, src, _ = p_res
                        threshold = config.get_bt_confidence_threshold(src)
                        if conf >= threshold:
                            bt_filters[needs_bt_or_ai_indices[pos]] = bt_tag

        for i, cat_indices in pending_exact.items():
            conf_bt = bt_filters[i]
            selected_idx = cat_indices[0]
            if conf_bt:
                conf_bt_norm = conf_bt.strip().lower()
                bt_matched = [idx for idx in cat_indices if cat_bts_lower[idx] == conf_bt_norm]
                if bt_matched:
                    selected_idx = bt_matched[0]
            bypass_results[i] = (
                cat_records[selected_idx],
                1.0,
                "High Confidence",
                "Exact Text Match",
            )

        for i, cand_indices in pending_fuzzy.items():
            conf_bt = bt_filters[i]
            valid_cands = cand_indices
            if conf_bt:
                conf_bt_norm = conf_bt.strip().lower()
                bt_matched = [idx for idx in cand_indices if cat_bts_lower[idx] == conf_bt_norm]
                if bt_matched:
                    valid_cands = bt_matched
            if valid_cands:
                selected_idx, weight_reason = resolve_weight_bypass_candidate(
                    cat_records, valid_cands, input_w_data_list[i]
                )
                bypass_results[i] = (
                    cat_records[selected_idx],
                    1.0,
                    "High Confidence",
                    f"Fuzzy Match (100%){weight_reason}",
                )
            else:
                ai_indices.append(i)
        if pending_fuzzy:
            ai_indices.sort()

        if progress_callback:
            progress_callback(10.0)

        # Step 2: Batch NER
        ai_entities = {}
        all_ner_texts = []
        for i in range(len(raw_names)):
            full_txt = TextPipeline.build_ner_input(raw_names[i], descriptions[i], categories[i])
            all_ner_texts.append(TextPipeline.prep_for_ner(full_txt))

        if self.ner and all_ner_texts:
            extracted_entities = self.ner.batch_extract_entities(all_ner_texts)
            for i, ents in enumerate(extracted_entities):
                ai_entities[i] = ents
        elif all_ner_texts:
            for i in range(len(all_ner_texts)):
                ai_entities[i] = {}

        if progress_callback:
            progress_callback(30.0)

        # Step 3: Final Matching Loop (Unrolled for Batching)
        logger.info(f"[MATCH] Finalizing semantic retrieval matches for {len(raw_names)} items...")

        # 3B. Batch Qdrant Queries
        search_dense = []
        search_bm25 = []
        search_sparse = []
        search_filters = []
        search_map = []  # stores (sku_index, query_type) e.g. (i, 'filtered'), (i, 'unfiltered'), (i, 'stripped')

        # Pre-batch brand stripped embeddings (skip no-op queries where stripped == clean_input)
        stripped_batch_texts = []
        stripped_batch_descs = []
        stripped_batch_cats = []
        stripped_batch_indices = []

        for i in ai_indices:
            if (
                getattr(config, "ENABLE_BRAND_STRIPPED_SEARCH", True)
                and ai_entities.get(i)
                and any(ai_entities[i].get("brand", []))
            ):
                brand_stripped = clean_inputs[i]
                for b in ai_entities[i].get("brand", []):
                    pattern = re.compile(r"\b" + re.escape(b) + r"\b", re.IGNORECASE)
                    brand_stripped = pattern.sub("", brand_stripped)
                brand_stripped = " ".join(brand_stripped.split()).strip()
                if brand_stripped and brand_stripped != clean_inputs[i]:
                    stripped_batch_texts.append(brand_stripped)
                    stripped_batch_descs.append(descriptions[i])
                    stripped_batch_cats.append(categories[i])
                    stripped_batch_indices.append(i)

        stripped_embs_dense = {}
        stripped_embs_bm25 = {}
        stripped_embs_sparse = {}
        if stripped_batch_texts:
            try:
                stripped_encoded = self.embedder.embed_weighted_sku(
                    stripped_batch_texts,
                    stripped_batch_descs,
                    stripped_batch_cats,
                    weights=config.MATCHER_WEIGHTS,
                )
                str_bm25 = stripped_encoded.get("bm25") or []
                str_sparse = stripped_encoded.get("sparse") or []
                for idx, orig_idx in enumerate(stripped_batch_indices):
                    stripped_embs_dense[orig_idx] = stripped_encoded["dense"][idx]
                    stripped_embs_bm25[orig_idx] = str_bm25[idx] if idx < len(str_bm25) else {}
                    stripped_embs_sparse[orig_idx] = (
                        str_sparse[idx] if idx < len(str_sparse) else None
                    )
            except Exception as e:
                logger.warning(f"Batch brand-stripped search setup failed: {e}")

        for i in ai_indices:
            dense = input_dense_embs[i]
            bm25 = input_bm25_embs.get(i, {})
            sparse = input_sparse_embs.get(i)

            # BT-filtered query
            if bt_filters[i]:
                search_dense.append(dense)
                search_bm25.append(bm25)
                search_sparse.append(sparse)
                search_filters.append(bt_filters[i])
                search_map.append((i, "filtered"))

            # Unfiltered query
            search_dense.append(dense)
            search_bm25.append(bm25)
            search_sparse.append(sparse)
            search_filters.append(None)
            search_map.append((i, "unfiltered"))

            # Brand-stripped query
            if i in stripped_embs_dense:
                search_dense.append(stripped_embs_dense[i])
                search_bm25.append(stripped_embs_bm25.get(i, {}))
                search_sparse.append(stripped_embs_sparse.get(i))
                search_filters.append(None)
                search_map.append((i, "stripped"))

        # Execute batch Qdrant searches — keep candidates as lightweight list[dict] (zero DataFrame churn)
        sku_candidates: dict[int, dict[str, list[dict[str, Any]]]] = {
            i: {"filtered": [], "unfiltered": [], "stripped": []} for i in ai_indices
        }
        if search_dense:
            all_candidates = self.vector_store.search_batch_from_vectors(
                search_dense,
                search_sparse,
                search_filters,
                domain=self.domain,
                bm25_vec_dicts=search_bm25,
            )
            for res_idx, candidates_list in enumerate(all_candidates):
                sku_idx, q_type = search_map[res_idx]
                sku_candidates[sku_idx][q_type] = candidates_list or []

        if progress_callback:
            progress_callback(50.0)

        # 3C. Batch Cross-Encoder Scoring
        cross_pairs = []
        pair_map = []  # stores (sku_index, candidate_key)

        for i in ai_indices:
            clean_in_no_w = input_no_weights_list[i]
            in_ents = ai_entities.get(i) or {}
            in_bf_key = "flavor" if self.domain == config.DOMAIN_FOOD else "brand"
            in_brand_or_flavor = ", ".join(sorted(str(x) for x in in_ents.get(in_bf_key, []) if x))
            formatted_query = TextPipeline.format_matcher_rerank_text(
                name=clean_in_no_w,
                seller_category=categories[i],
                brand_or_flavor=in_brand_or_flavor,
                description=descriptions[i],
                domain=self.domain,
            )

            all_cands_list: list[dict[str, Any]] = []
            for q_type in ("filtered", "unfiltered", "stripped"):
                cands_for_type = sku_candidates[i][q_type]
                if cands_for_type:
                    all_cands_list.extend(cands_for_type)

            all_cands_list.sort(
                key=lambda c: (-c.get("_qdrant_score_", 0.0), c.get("clean_text", ""))
            )

            seen_keys = set()
            rerank_limit = getattr(config, "MATCHER_RERANK_CANDIDATES", 30)
            for cand_row in all_cands_list:
                if len(seen_keys) >= rerank_limit:
                    break
                cand_key = cand_row.get("db_uid") or cand_row.get("clean_text", "")
                if cand_key not in seen_keys:
                    seen_keys.add(cand_key)
                    formatted_cand = TextPipeline.format_catalog_row_for_reranker(
                        cand_row, domain=self.domain
                    )
                    cross_pairs.append([formatted_query, formatted_cand])
                    pair_map.append((i, cand_key))

        cross_scores_list = []
        if cross_pairs:
            cross_scores_list = self.embedder.score_cross_encoder(cross_pairs)

        if progress_callback:
            progress_callback(75.0)

        sku_scored_cands: dict[int, dict[Any, float]] = {i: {} for i in ai_indices}
        for pair_idx, (sku_idx, cand_key) in enumerate(pair_map):
            sku_scored_cands[sku_idx][cand_key] = float(cross_scores_list[pair_idx])

        def score_candidates(
            cands: list[dict[str, Any]], sku_idx: int
        ) -> list[tuple[float, dict[str, Any]]]:
            if not cands:
                return []
            s_map = sku_scored_cands[sku_idx]
            scored = [
                (float(s_map.get(c.get("db_uid") or c.get("clean_text", ""), -10.0)), c)
                for c in cands
            ]
            scored.sort(
                key=lambda item: (
                    -float(np.round(item[0], 3)),
                    str(item[1].get("clean_text", "")),
                )
            )
            return scored

        # 3D. Final Evaluation Loop
        logger.info(
            f"[MATCH] Executing final business rules evaluation for {len(raw_names)} SKUs..."
        )
        gate_limit = config.MATCHER_LOGIC_GATE_CANDIDATES
        for i in range(len(raw_names)):
            if progress_callback:
                progress_callback(
                    75.0 + (((i + 1) / len(raw_names)) * 25.0),
                    f"Matching {i + 1} of {len(raw_names)}...",
                )

            best_match_row_fallback = None
            final_score_fallback = -10.0
            status_fallback = "Rejected"
            reasons_fallback = ""

            raw_input, clean_input, input_no_weights = (
                raw_names[i],
                clean_inputs[i],
                input_no_weights_list[i],
            )
            current_input_price = prices[i]
            input_description = descriptions[i]
            input_category = categories[i]

            if bypass_results[i]:
                best_match_row, final_score, status, reasons = bypass_results[i]
                input_entities = ai_entities.get(i)
                input_w_data = input_w_data_list[i]
            else:
                input_entities, input_w_data = ai_entities[i], input_w_data_list[i]
                bt_filter = bt_filters[i]

                best_match_row = None
                final_score = -10.0
                status = "Rejected"
                reasons = "No candidates found"
                bt_used = False

                # 1. Attempt BT-filtered search
                if bt_filter:
                    candidates_filtered = score_candidates(sku_candidates[i]["filtered"], i)
                    if candidates_filtered:
                        best_match_row_f = None
                        final_score_f = -10.0
                        status_f = "Rejected"
                        reasons_f = ""
                        best_cand_score_f = -100.0
                        for cand_score, cand_row in candidates_filtered[:gate_limit]:
                            f_score, f_status, f_reasons = self.rules.apply_logic_gates(
                                clean_input,
                                input_entities,
                                cand_row,
                                cand_score,
                                current_input_price,
                                input_w_data,
                                input_no_weights,
                                domain=self.domain,
                                input_description=input_description,
                                input_category=input_category,
                                predicted_bt=bt_filter,
                            )

                            is_better = False
                            if best_match_row_f is None:
                                is_better = True
                            elif f_score > final_score_f + 1e-4:
                                is_better = True
                            elif abs(f_score - final_score_f) <= 1e-4:
                                if cand_score > best_cand_score_f + 1e-3:
                                    is_better = True
                                elif abs(cand_score - best_cand_score_f) <= 1e-3:
                                    curr_text = str(
                                        cand_row.get("clean_text", cand_row.get("Name", ""))
                                    )
                                    best_text = str(
                                        best_match_row_f.get(
                                            "clean_text", best_match_row_f.get("Name", "")
                                        )
                                    )
                                    if curr_text < best_text:
                                        is_better = True

                            if is_better:
                                best_match_row_f = cand_row
                                final_score_f = f_score
                                status_f = f_status
                                reasons_f = f_reasons
                                best_cand_score_f = cand_score

                        if status_f == "High Confidence":
                            best_match_row = best_match_row_f
                            final_score = final_score_f
                            status = status_f
                            reasons = f"BT-Filtered ({bt_filter}) | {reasons_f}"
                            bt_used = True
                        else:
                            best_match_row_fallback = best_match_row_f
                            final_score_fallback = final_score_f
                            status_fallback = status_f
                            reasons_fallback = (
                                f"BT-Filtered ({bt_filter}) [Low Confidence Fallback] | {reasons_f}"
                            )

                # 2. Fallback to Full-Catalog Search
                if not bt_used:
                    cands_unf = sku_candidates[i]["unfiltered"]
                    cands_str = sku_candidates[i]["stripped"]
                    dedup_unfiltered: list[dict[str, Any]] = []
                    if cands_unf or cands_str:
                        seen_unf: set[Any] = set()
                        for cand in (*cands_unf, *cands_str):
                            ct = cand.get("clean_text")
                            if ct not in seen_unf:
                                seen_unf.add(ct)
                                dedup_unfiltered.append(cand)

                    if not dedup_unfiltered:
                        if best_match_row_fallback is not None:
                            best_match_row = best_match_row_fallback
                            final_score = final_score_fallback
                            status = status_fallback
                            reasons = reasons_fallback
                        else:
                            results.append(
                                {
                                    "Input Raw": raw_input,
                                    "Matched Catalog Name": "",
                                    "Final Score": 0.0,
                                    "Status": "Low / Rejected",
                                    "Logic Notes": "No candidates found",
                                    "BasicType": "",
                                    "GenericKeywords": "",
                                    "Categories": "",
                                    "Region": "",
                                    "Input Entities": input_entities,
                                    "Catalog Entities": {},
                                }
                            )
                            continue
                    else:
                        candidates_unfiltered = score_candidates(dedup_unfiltered, i)
                        best_match_row_uf = None
                        final_score_uf = -10.0
                        status_uf = "Rejected"
                        reasons_uf = ""
                        best_cand_score_uf = -100.0

                        for cand_score, cand_row in candidates_unfiltered[:gate_limit]:
                            uf_score, uf_status, uf_reasons = self.rules.apply_logic_gates(
                                clean_input,
                                input_entities,
                                cand_row,
                                cand_score,
                                current_input_price,
                                input_w_data,
                                input_no_weights,
                                domain=self.domain,
                                input_description=input_description,
                                input_category=input_category,
                                predicted_bt=bt_filter,
                            )

                            is_better = False
                            if best_match_row_uf is None:
                                is_better = True
                            elif uf_score > final_score_uf + 1e-4:
                                is_better = True
                            elif abs(uf_score - final_score_uf) <= 1e-4:
                                if cand_score > best_cand_score_uf + 1e-3:
                                    is_better = True
                                elif abs(cand_score - best_cand_score_uf) <= 1e-3:
                                    curr_text = str(
                                        cand_row.get("clean_text", cand_row.get("Name", ""))
                                    )
                                    best_text = str(
                                        best_match_row_uf.get(
                                            "clean_text", best_match_row_uf.get("Name", "")
                                        )
                                    )
                                    if curr_text < best_text:
                                        is_better = True

                            if is_better:
                                best_match_row_uf = cand_row
                                final_score_uf = uf_score
                                status_uf = uf_status
                                reasons_uf = uf_reasons
                                best_cand_score_uf = cand_score

                        if (
                            best_match_row_fallback is not None
                            and final_score_fallback > final_score_uf
                        ):
                            best_match_row = best_match_row_fallback
                            final_score = final_score_fallback
                            status = status_fallback
                            reasons = reasons_fallback
                        else:
                            best_match_row = best_match_row_uf
                            final_score = final_score_uf
                            status = status_uf
                            if bt_filter:
                                reasons = f"BT-Fallback (BT: {bt_filter}, score: {final_score_uf:.4f}) | {reasons_uf}"
                            else:
                                reasons = reasons_uf

            # Build standard result dictionary
            match_gk = best_match_row.get("Generic keywords", "")
            if self.domain == config.DOMAIN_FOOD and getattr(self, "flavor_categories", None):
                # 1. Resolve input flavors
                input_meats, input_seafoods, input_vegs = triage_input_flavors(
                    self.rules, self.flavor_categories, input_entities, best_match_row
                )

                if match_gk:
                    kws = [k.strip() for k in str(match_gk).split(",") if k.strip()]
                    filtered_kws = []
                    for kw in kws:
                        kw_lower = kw.lower()

                        # A. Check generic/literal keywords using dynamic term sets
                        # Seafood literal check: allowed if input name has a seafood term OR input has at least one seafood flavor
                        if self._seafood_pattern and self._seafood_pattern.search(kw_lower):
                            has_input_seafood_term = bool(self._seafood_pattern.search(clean_input))
                            if not (has_input_seafood_term or input_seafoods):
                                continue

                        # Mixed literal check: allowed only if input name has a mixed term OR multiple meat/seafood flavors, with at least one non-seafood meat
                        elif self._mixed_pattern and self._mixed_pattern.search(kw_lower):
                            has_input_mixed_term = bool(self._mixed_pattern.search(clean_input))
                            total_unique_items = len(input_meats) + len(input_seafoods)
                            is_mixed_by_flavors = total_unique_items >= 2 and len(input_meats) >= 1
                            if not (has_input_mixed_term or is_mixed_by_flavors):
                                continue

                        # Vegetable literal check: allowed if input name has a veg term OR input has at least one vegetable flavor
                        elif self._veg_pattern and self._veg_pattern.search(kw_lower):
                            has_input_veg_term = bool(self._veg_pattern.search(clean_input))
                            if not (has_input_veg_term or input_vegs):
                                continue

                        # B. Specific flavor terms check
                        keep_kw = True
                        if self._specific_flavor_pattern:
                            for match in self._specific_flavor_pattern.finditer(kw_lower):
                                term = match.group(1).lower()
                                is_meat, is_veg, is_seafood = self.flavor_categories.get(
                                    term, (False, False, False)
                                )
                                canonical = (
                                    self.rules.food_flavors_dict.get(term, term)
                                    if hasattr(self.rules, "food_flavors_dict")
                                    else term
                                )
                                if is_seafood:
                                    if canonical not in input_seafoods:
                                        keep_kw = False
                                        break
                                elif is_meat:
                                    if canonical not in input_meats:
                                        keep_kw = False
                                        break
                                elif is_veg:
                                    if canonical not in input_vegs:
                                        keep_kw = False
                                        break

                        if keep_kw:
                            filtered_kws.append(kw)
                    match_gk = ", ".join(filtered_kws)
            else:
                # Fallback/Market Domain logic: filter out all flavors (original behavior)
                flavors = set()
                if input_entities and isinstance(input_entities, dict):
                    flavors.update(x.lower() for x in input_entities.get("flavor", set()) if x)
                cat_ents = best_match_row.get("entities")
                if isinstance(cat_ents, dict):
                    flavors.update(x.lower() for x in cat_ents.get("flavor", set()) if x)

                if flavors and match_gk:
                    kws = [k.strip() for k in str(match_gk).split(",") if k.strip()]
                    filtered_kws = []
                    for kw in kws:
                        kw_lower = kw.lower()
                        if any(f in kw_lower for f in flavors):
                            continue
                        filtered_kws.append(kw)
                    match_gk = ", ".join(filtered_kws)

            results.append(
                {
                    "Input Raw": raw_input,
                    "Matched Catalog Name": best_match_row.get("Name", ""),
                    "Final Score": round(float(final_score), 4),
                    "Status": status,
                    "Logic Notes": reasons,
                    "BasicType": best_match_row.get("basictype", ""),
                    "GenericKeywords": match_gk,
                    "Categories": best_match_row.get("category", ""),
                    "Region": best_match_row.get("region", ""),
                    "Input Entities": input_entities,
                    "Catalog Entities": best_match_row.get("entities"),
                }
            )

        if progress_callback:
            progress_callback(100.0)

        return pd.DataFrame(results)
