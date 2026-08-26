import gc
import logging
import re
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm

from engine import config
from engine.nlp.text_cleaner import TextPipeline
from engine.utils.flavor_utils import build_food_flavors_info

logger = logging.getLogger("matchops.matcher")

class SKUMatcher:
    """Main SKU matching pipeline orchestrator."""

    def __init__(self, catalog_df: pd.DataFrame, brands_df: pd.DataFrame, ner_engine, embed_engine, cache_manager, logic_gates, domain: str = config.DOMAIN_MARKET, classifier=None):
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
                    if aliases_str and aliases_str.lower() != "none" and aliases_str.lower() != "nan":
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
            sorted_t = sorted([t.lower().strip() for t in terms_set if t.strip()], key=len, reverse=True)
            if not sorted_t:
                return None
            return re.compile(r"\b(" + "|".join(re.escape(t) for t in sorted_t) + r")\b", re.IGNORECASE)

        self._mixed_pattern = _compile_terms(self.mixed_terms)
        self._seafood_pattern = _compile_terms(self.seafood_terms)
        self._veg_pattern = _compile_terms(self.veg_terms)

        if self.flavor_categories:
            specific_terms = [
                term for term in self.flavor_categories.keys()
                if term not in self.mixed_terms and term not in self.seafood_terms and term not in self.veg_terms
            ]
            self._specific_flavor_pattern = _compile_terms(specific_terms)
        else:
            self._specific_flavor_pattern = None

        # Sync catalog with cache and vector store
        self.raw_catalog, self.vector_store = cache_manager.manage_catalog_cache(catalog_df, brands_df, domain=domain)
        self.raw_catalog = self.raw_catalog.reset_index(drop=True)

        # Build lookup maps for fast O(1) exact matching and O(1) token-sort fuzzy bypass
        self.exact_match_map: Dict[str, int] = {}
        self.token_sorted_map: Dict[str, int] = {}

        for i, row in self.raw_catalog.iterrows():
            clean_txt = str(row.get("clean_text") or "").strip()
            if not clean_txt or clean_txt == "None":
                clean_txt = TextPipeline.normalize_final(TextPipeline.standardize_units(str(row.get("Name", ""))))
            no_weights = str(row.get("clean_no_weights") or TextPipeline.strip_weights(clean_txt)).strip()

            if clean_txt and clean_txt != "None" and clean_txt not in self.exact_match_map:
                self.exact_match_map[clean_txt] = i

            sorted_tokens = " ".join(sorted(no_weights.split())).strip()
            if sorted_tokens and sorted_tokens != "None" and sorted_tokens not in self.token_sorted_map:
                self.token_sorted_map[sorted_tokens] = i

    def search_candidates(self, input_vec_dense: np.ndarray, input_vec_sparse: Dict[str, float], bt_filter: Optional[str] = None) -> pd.DataFrame:
        """Retrieves top-K candidates from the vector store using hybrid search, optionally filtered by BT."""
        hits = self.vector_store.search_from_vectors(
            dense_vec=input_vec_dense, sparse_vec_dict=input_vec_sparse, domain=self.domain, top_k=config.TOP_K_RETRIEVAL, bt_filter=bt_filter
        )
        return pd.DataFrame(hits) if hits else pd.DataFrame()

    def process_inputs(self, input_df: pd.DataFrame, progress_callback: Optional[Callable] = None, chunk_size: Optional[int] = None) -> pd.DataFrame:
        """Matches input SKUs against the catalog with automatic chunking to keep RAM flat and prevent OOM."""
        total_skus = len(input_df)
        if total_skus == 0:
            return pd.DataFrame()

        effective_chunk_size = chunk_size or getattr(config, "MATCH_CHUNK_SIZE", 250)

        if total_skus <= effective_chunk_size:
            return self._process_single_chunk(input_df, progress_callback=progress_callback)

        num_chunks = (total_skus + effective_chunk_size - 1) // effective_chunk_size
        logger.info(f"[MATCH] Processing {total_skus} SKUs in {num_chunks} chunks (chunk_size={effective_chunk_size})...")

        results = []
        for chunk_idx, chunk_start in enumerate(range(0, total_skus, effective_chunk_size)):
            chunk_end = min(chunk_start + effective_chunk_size, total_skus)
            chunk_df = input_df.iloc[chunk_start:chunk_end]

            def chunk_cb(pct, msg=None):
                if progress_callback:
                    overall_pct = ((chunk_start + (float(pct) / 100.0) * len(chunk_df)) / total_skus) * 100.0
                    progress_callback(overall_pct, msg or f"Chunk {chunk_idx + 1}/{num_chunks} ({chunk_end}/{total_skus} SKUs)")

            chunk_res = self._process_single_chunk(chunk_df, progress_callback=chunk_cb)
            results.append(chunk_res)
            gc.collect()

        if progress_callback:
            progress_callback(100.0, "Matching completed.")

        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    def _process_single_chunk(self, input_df: pd.DataFrame, progress_callback: Optional[Callable] = None) -> pd.DataFrame:
        """Matches a single chunk of input SKUs against the catalog."""
        results = []
        logger.info(f"[MATCH] Pre-processing {len(input_df)} input SKUs...")

        raw_names, clean_inputs, prices, input_no_weights_list, input_w_data_list = [], [], [], [], []
        for _, row in input_df.iterrows():
            name = str(row.get("Name", row.iloc[0]))
            raw_names.append(name)

            clean_input = TextPipeline.normalize_final(TextPipeline.standardize_units(name))
            clean_inputs.append(clean_input)
            input_no_weights_list.append(TextPipeline.strip_weights(clean_input))
            input_w_data_list.append(TextPipeline.extract_weight_feature(clean_input))

            # Extract price for logic gate validation
            try:
                raw_price = row.get("Price", 0)
                prices.append(float(re.sub(r"[^\d\.]", "", str(raw_price))) if raw_price else 0.0)
            except (ValueError, TypeError):
                prices.append(0.0)

        # Step 1: Exact & Fuzzy Matching (AI Bypass)
        ai_indices = []
        bypass_results: List[Optional[Tuple]] = [None] * len(raw_names)
        logger.info("[MATCH] [Step 1] Running AI bypass matching...")

        for i, clean_input in enumerate(clean_inputs):
            input_no_weights = input_no_weights_list[i]
            input_w_data = input_w_data_list[i]

            if not clean_input or clean_input == "None":
                ai_indices.append(i)
                continue

            # Exact text match lookup
            cat_idx = self.exact_match_map.get(clean_input)
            if cat_idx is not None:
                bypass_results[i] = (self.raw_catalog.iloc[cat_idx], 100.0, "High Confidence", "Exact Text Match")
                continue

            # Early fuzzy search on weight-stripped text (O(1) Hash Map Optimization)
            sorted_input_tokens = " ".join(sorted(str(input_no_weights).split()))
            cat_idx_sorted = self.token_sorted_map.get(sorted_input_tokens)

            if cat_idx_sorted is not None:
                c_idx = cat_idx_sorted
                cat_row = self.raw_catalog.iloc[c_idx]
                catalog_w_data = cat_row.get("weight_val")

                combined_score, weight_reason = 100.0, ""
                weight_mismatch = False

                # Weight logic boost/penalty
                if input_w_data[0] is not None and catalog_w_data is not None and isinstance(catalog_w_data, (tuple, list)) and catalog_w_data[0] is not None:
                    in_val, _, in_type = input_w_data
                    cat_val, _, cat_type = catalog_w_data
                    if in_type == cat_type:
                        max_val = max(in_val, cat_val)
                        diff_pct = abs(in_val - cat_val) / max_val * 100 if max_val > 0 else 0
                        if diff_pct < 1.0:
                            combined_score = min(100.0, combined_score + 2.0)
                            weight_reason = f" | Weight Match ({int(in_val)})"
                        elif diff_pct > 10.0:
                            weight_mismatch = True

                if weight_mismatch:
                    # Do not bypass if weights mismatch (e.g. single item vs multipack); pass to AI retrieval
                    ai_indices.append(i)
                else:
                    best_reason = f"Fuzzy Match (100%){weight_reason}"
                    bypass_results[i] = (self.raw_catalog.iloc[c_idx], combined_score, "High Confidence", best_reason)
            else:
                ai_indices.append(i)

        if progress_callback:
            progress_callback(10.0)

        # Step 2: Batch AI Operations (Retrieval + NER)
        ai_entities, input_dense_embs, input_sparse_embs = {}, [None] * len(raw_names), [None] * len(raw_names)
        if ai_indices:
            logger.info(f"[MATCH] [Step 2] Batch-encoding {len(ai_indices)} SKUs for retrieval...")
            ai_clean_inputs = [clean_inputs[i] for i in ai_indices]
            ai_descriptions = [str(input_df.iloc[i].get("Description", input_df.iloc[i].get("description", ""))) for i in ai_indices]
            ai_categories = [str(input_df.iloc[i].get("Category", input_df.iloc[i].get("category", ""))) for i in ai_indices]
            encoded = self.embedder.embed_weighted_sku(
                ai_clean_inputs, ai_descriptions, ai_categories,
                weights=config.MATCHER_WEIGHTS
            )
            for list_idx, original_idx in enumerate(ai_indices):
                input_dense_embs[original_idx] = encoded["dense"][list_idx]
                input_sparse_embs[original_idx] = encoded["sparse"][list_idx]

            logger.info(f"[MATCH] [Step 3] Running batch NER for {len(ai_indices)} SKUs...")

        # Batch NER for all input SKUs (including bypass matches and taking description/category into account)
        all_ner_texts = []
        for i in range(len(raw_names)):
            i_desc = str(input_df.iloc[i].get("Description", input_df.iloc[i].get("description", "")))
            i_cat = str(input_df.iloc[i].get("Category", input_df.iloc[i].get("category", "")))
            full_txt = raw_names[i]
            if i_desc and i_desc.lower() not in ("nan", "none", "<na>"):
                full_txt += f" {i_desc}"
            if i_cat and i_cat.lower() not in ("nan", "none", "<na>"):
                full_txt += f" {i_cat}"
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
        logger.info(f"[MATCH] [Step 4] Finalizing matches for {len(raw_names)} items...")
        
        # 3A. Batch Predict BT for missing items
        bt_filters = [None] * len(raw_names)
        if self.classifier and ai_indices:
            ai_dense_vecs = np.array([input_dense_embs[i] for i in ai_indices])
            ai_prices = [prices[i] for i in ai_indices]
            bt_preds = self.classifier.batch_predict_bt(ai_dense_vecs, ai_prices)
            for list_idx, ai_idx in enumerate(ai_indices):
                if bt_preds[list_idx]:
                    bt_tag, confidence, source = bt_preds[list_idx]
                    threshold = 0.40 if source == "zero-shot" else 0.50
                    if confidence >= threshold:
                        bt_filters[ai_idx] = bt_tag

        # 3B. Batch Qdrant Queries
        search_dense = []
        search_sparse = []
        search_filters = []
        search_map = [] # stores (sku_index, query_type) e.g. (i, 'filtered'), (i, 'unfiltered'), (i, 'stripped')
        
        # Pre-batch brand stripped embeddings
        stripped_batch_texts = []
        stripped_batch_descs = []
        stripped_batch_cats = []
        stripped_batch_indices = []
        
        for i in ai_indices:
            if getattr(config, "ENABLE_BRAND_STRIPPED_SEARCH", True) and ai_entities.get(i) and any(ai_entities[i].get("brand", [])):
                brand_stripped = clean_inputs[i]
                for b in ai_entities[i].get("brand", []):
                    pattern = re.compile(r"\b" + re.escape(b) + r"\b", re.IGNORECASE)
                    brand_stripped = pattern.sub("", brand_stripped)
                brand_stripped = " ".join(brand_stripped.split()).strip()
                if brand_stripped:
                    row_data = input_df.iloc[i]
                    i_desc = str(row_data.get("Description", row_data.get("description", "")))
                    i_cat = str(row_data.get("Category", row_data.get("category", "")))
                    stripped_batch_texts.append(brand_stripped)
                    stripped_batch_descs.append(i_desc)
                    stripped_batch_cats.append(i_cat)
                    stripped_batch_indices.append(i)
        
        stripped_embs_dense = {}
        stripped_embs_sparse = {}
        if stripped_batch_texts:
            try:
                stripped_encoded = self.embedder.embed_weighted_sku(
                    stripped_batch_texts, stripped_batch_descs, stripped_batch_cats, weights=config.MATCHER_WEIGHTS
                )
                for idx, orig_idx in enumerate(stripped_batch_indices):
                    stripped_embs_dense[orig_idx] = stripped_encoded["dense"][idx]
                    stripped_embs_sparse[orig_idx] = stripped_encoded["sparse"][idx]
            except Exception as e:
                logger.warning(f"Batch brand-stripped search setup failed: {e}")

        for i in ai_indices:
            dense, sparse = input_dense_embs[i], input_sparse_embs[i]
            
            # BT-filtered query
            if bt_filters[i]:
                search_dense.append(dense)
                search_sparse.append(sparse)
                search_filters.append(bt_filters[i])
                search_map.append((i, 'filtered'))
            
            # Unfiltered query
            search_dense.append(dense)
            search_sparse.append(sparse)
            search_filters.append(None)
            search_map.append((i, 'unfiltered'))
            
            # Brand-stripped query
            if i in stripped_embs_dense:
                search_dense.append(stripped_embs_dense[i])
                search_sparse.append(stripped_embs_sparse[i])
                search_filters.append(None)
                search_map.append((i, 'stripped'))
        
        # Execute batch Qdrant searches
        sku_candidates = {i: {'filtered': pd.DataFrame(), 'unfiltered': pd.DataFrame(), 'stripped': pd.DataFrame()} for i in ai_indices}
        if search_dense:
            all_candidates = self.vector_store.search_batch_from_vectors(
                search_dense, search_sparse, search_filters, domain=self.domain
            )
            for res_idx, candidates_list in enumerate(all_candidates):
                sku_idx, q_type = search_map[res_idx]
                # Convert list of dicts to DataFrame
                df_cands = pd.DataFrame(candidates_list) if candidates_list else pd.DataFrame()
                sku_candidates[sku_idx][q_type] = df_cands

        if progress_callback:
            progress_callback(50.0)
            
        # 3C. Batch Cross-Encoder Scoring
        cross_pairs = []
        pair_map = [] # stores (sku_index, q_type, cand_idx, clean_text)
        
        for i in ai_indices:
            clean_in = clean_inputs[i]
            
            # Combine all candidates for this SKU and sort by Qdrant score
            all_cands_list = []
            for q_type in ['filtered', 'unfiltered', 'stripped']:
                df_cands = sku_candidates[i][q_type]
                if not df_cands.empty:
                    for cand_idx, cand_row in df_cands.iterrows():
                        all_cands_list.append((q_type, cand_idx, cand_row))
            
            all_cands_list.sort(key=lambda x: x[2].get("_qdrant_score_", 0.0), reverse=True)
            
            seen_texts = set()
            for q_type, cand_idx, cand_row in all_cands_list:
                if len(seen_texts) >= 15:
                    break
                txt = cand_row.get("clean_text", "")
                if txt not in seen_texts:
                    seen_texts.add(txt)
                    cross_pairs.append([clean_in, txt])
                    pair_map.append((i, q_type, cand_idx, txt))
                        
        cross_scores_list = []
        if cross_pairs:
            cross_scores_list = self.embedder.score_cross_encoder(cross_pairs)

        if progress_callback:
            progress_callback(75.0)
            
        # Map scores back to candidate dataframes
        # We will add a 'cross_score' column to the candidate dataframes
        sku_scored_cands = {i: {} for i in ai_indices}
        for pair_idx, (sku_idx, q_type, cand_idx, txt) in enumerate(pair_map):
            sku_scored_cands[sku_idx][txt] = cross_scores_list[pair_idx]
            
        # Helper to score a candidate dataframe
        def score_candidate_df(df_cands: pd.DataFrame, sku_idx: int, clean_in: str, ents: dict) -> pd.DataFrame:
            if df_cands.empty:
                return df_cands
            
            c_scores = np.array([sku_scored_cands[sku_idx].get(txt, -10.0) for txt in df_cands["clean_text"]])
            in_toks = len(clean_in.split())
            cat_toks = df_cands["token_count"].values if "token_count" in df_cands else np.zeros(len(df_cands))
            df_cands = df_cands.copy()
            df_cands['cross_score'] = c_scores
            return df_cands.sort_values(by='cross_score', ascending=False)
            
        # 3D. Final Evaluation Loop
        for i in tqdm(range(len(raw_names)), unit="sku"):
            if progress_callback:
                progress_callback(75.0 + (((i + 1) / len(raw_names)) * 25.0), f"Matching {i + 1} of {len(raw_names)}...")

            raw_input, clean_input, input_no_weights = raw_names[i], clean_inputs[i], input_no_weights_list[i]
            row_data = input_df.iloc[i]
            current_input_price = prices[i]
            input_description = str(row_data.get("Description", row_data.get("description", "")))
            input_category = str(row_data.get("Category", row_data.get("category", "")))

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
                    candidates_filtered = score_candidate_df(sku_candidates[i]['filtered'], i, clean_input, input_entities)
                    if not candidates_filtered.empty:
                        best_match_row_f = None
                        final_score_f = -10.0
                        status_f = "Rejected"
                        reasons_f = ""

                        for idx, cand_row in candidates_filtered.head(5).iterrows():
                            cand_score = cand_row['cross_score']
                            cand_row_copy = cand_row.drop('cross_score').copy()

                            f_score, f_status, f_reasons = self.rules.apply_logic_gates(
                                clean_input, input_entities, cand_row_copy, cand_score,
                                current_input_price, input_w_data, input_no_weights,
                                domain=self.domain, input_description=input_description,
                                input_category=input_category, predicted_bt=bt_filter
                            )

                            if best_match_row_f is None or f_score > final_score_f:
                                best_match_row_f = cand_row_copy
                                final_score_f = f_score
                                status_f = f_status
                                reasons_f = f_reasons

                            if f_status == "High Confidence":
                                best_match_row_f = cand_row_copy
                                final_score_f = f_score
                                status_f = f_status
                                reasons_f = f_reasons
                                break

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
                            reasons_fallback = f"BT-Filtered ({bt_filter}) [Low Confidence Fallback] | {reasons_f}"

                # 2. Fallback to Full-Catalog Search
                if not bt_used:
                    cands_unf = sku_candidates[i]['unfiltered']
                    cands_str = sku_candidates[i]['stripped']
                    candidates_unfiltered = pd.concat([cands_unf, cands_str], ignore_index=True) if not cands_str.empty else cands_unf
                    
                    if not candidates_unfiltered.empty:
                        candidates_unfiltered = candidates_unfiltered.drop_duplicates(subset=["clean_text"])
                    
                    if candidates_unfiltered.empty:
                        if 'best_match_row_fallback' in locals():
                            best_match_row = best_match_row_fallback
                            final_score = final_score_fallback
                            status = status_fallback
                            reasons = reasons_fallback
                        else:
                            continue
                    else:
                        candidates_unfiltered = score_candidate_df(candidates_unfiltered, i, clean_input, input_entities)
                        best_match_row_uf = None
                        final_score_uf = -10.0
                        status_uf = "Rejected"
                        reasons_uf = ""

                        for idx, cand_row in candidates_unfiltered.head(5).iterrows():
                            cand_score = cand_row['cross_score']
                            cand_row_copy = cand_row.drop('cross_score').copy()

                            uf_score, uf_status, uf_reasons = self.rules.apply_logic_gates(
                                clean_input, input_entities, cand_row_copy, cand_score,
                                current_input_price, input_w_data, input_no_weights,
                                domain=self.domain, input_description=input_description,
                                input_category=input_category, predicted_bt=bt_filter
                            )

                            if best_match_row_uf is None or uf_score > final_score_uf:
                                best_match_row_uf = cand_row_copy
                                final_score_uf = uf_score
                                status_uf = uf_status
                                reasons_uf = uf_reasons

                            if uf_status == "High Confidence":
                                best_match_row_uf = cand_row_copy
                                final_score_uf = uf_score
                                status_uf = uf_status
                                reasons_uf = uf_reasons
                                break

                        if 'best_match_row_fallback' in locals() and final_score_fallback > final_score_uf:
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
                input_flavors = set()
                if input_entities and isinstance(input_entities, dict):
                    input_flavors.update(x.lower() for x in input_entities.get("flavor", set()) if x)
                # Fallback to matched catalog entities if input_entities is empty/None (e.g. for exact match bypass)
                if not input_flavors:
                    cat_ents = best_match_row.get("entities")
                    if isinstance(cat_ents, dict):
                        input_flavors.update(x.lower() for x in cat_ents.get("flavor", set()) if x)
                resolved_input_flavors = self.rules._resolve_flavors(input_flavors) if hasattr(self.rules, "_resolve_flavors") else input_flavors
                
                input_meats = {f for f in resolved_input_flavors if self.flavor_categories.get(f, (False, False, False))[0] and not self.flavor_categories.get(f, (False, False, False))[2]}
                input_seafoods = {f for f in resolved_input_flavors if self.flavor_categories.get(f, (False, False, False))[2]}
                input_vegs = {f for f in resolved_input_flavors if self.flavor_categories.get(f, (False, False, False))[1]}
                
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
                            is_mixed_by_flavors = (total_unique_items >= 2 and len(input_meats) >= 1)
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
                                is_meat, is_veg, is_seafood = self.flavor_categories.get(term, (False, False, False))
                                canonical = self.rules.food_flavors_dict.get(term, term) if hasattr(self.rules, "food_flavors_dict") else term
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

            results.append({
                "Input Raw": raw_input, "Matched Catalog Name": best_match_row.get("Name", ""),
                "Final Score": round(float(final_score), 4), "Status": status, "Logic Notes": reasons,
                "BasicType": best_match_row.get("basictype", ""),
                "GenericKeywords": match_gk,
                "Categories": best_match_row.get("category", ""),
                "Region": best_match_row.get("region", ""),
                "Input Entities": input_entities, "Catalog Entities": best_match_row.get("entities")
            })

        if progress_callback:
            progress_callback(100.0)

        return pd.DataFrame(results)