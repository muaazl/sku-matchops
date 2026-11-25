import logging
import re
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from engine.core import config
from engine.preprocessing.flavor_preprocessor import build_food_flavors_info, triage_input_flavors
from engine.preprocessing.text_normalizer import (
    build_ner_input_text,
    extract_weight_feature,
    format_catalog_row_for_reranker,
    format_matcher_rerank_text,
    normalize_sku_text,
    prepare_ner_text,
    standardize_sku_units,
    strip_weight_tokens,
)
from engine.preprocessing.weight_preprocessor import (
    resolve_bt_consensus_indices,
    resolve_weight_bypass_candidate,
)

logger = logging.getLogger("matchops.matcher")


def is_better_gated_candidate(
    cand_score: float,
    cand_raw_score: float,
    cand_clean_text: str,
    best_score: float | None,
    best_raw_score: float | None,
    best_clean_text: str | None,
) -> bool:
    """Deterministic tie-breaking comparator across gated match candidates."""
    if best_score is None or best_raw_score is None or best_clean_text is None:
        return True
    if cand_score > best_score + 1e-4:
        return True
    if abs(cand_score - best_score) <= 1e-4:
        if cand_raw_score > best_raw_score + 1e-3:
            return True
        if abs(cand_raw_score - best_raw_score) <= 1e-3:
            return cand_clean_text < best_clean_text
    return False


def filter_matched_generic_keywords(
    match_gk: str,
    domain: str,
    clean_input: str,
    input_entities: dict | None,
    winner_row: dict | Any,
    rules: Any = None,
    flavor_categories: dict | None = None,
    seafood_pattern: Any = None,
    mixed_pattern: Any = None,
    veg_pattern: Any = None,
    specific_flavor_pattern: Any = None,
) -> str:
    """Filters mismatched flavor/generic keywords from a matched catalog SKU."""
    if not match_gk:
        return ""

    if domain == config.DOMAIN_FOOD and flavor_categories:
        input_meats, input_seafoods, input_vegs = triage_input_flavors(
            rules, flavor_categories, input_entities, winner_row
        )
        kws = [k.strip() for k in str(match_gk).split(",") if k.strip()]
        filtered_kws = []
        food_flavors = getattr(rules, "food_flavors_dict", {}) if rules is not None else {}

        for kw in kws:
            kw_lower = kw.lower()

            if seafood_pattern and seafood_pattern.search(kw_lower):
                if not (bool(seafood_pattern.search(clean_input)) or input_seafoods):
                    continue
            elif mixed_pattern and mixed_pattern.search(kw_lower):
                total_unique = len(input_meats) + len(input_seafoods)
                if not (
                    bool(mixed_pattern.search(clean_input))
                    or (total_unique >= 2 and len(input_meats) >= 1)
                ):
                    continue
            elif veg_pattern and veg_pattern.search(kw_lower):
                if not (bool(veg_pattern.search(clean_input)) or input_vegs):
                    continue

            keep_kw = True
            if specific_flavor_pattern:
                for match in specific_flavor_pattern.finditer(kw_lower):
                    term = match.group(1).lower()
                    is_meat, is_veg, is_seafood = flavor_categories.get(term, (False, False, False))
                    canonical = food_flavors.get(term, term)
                    if is_seafood and canonical not in input_seafoods:
                        keep_kw = False
                        break
                    if is_meat and canonical not in input_meats:
                        keep_kw = False
                        break
                    if is_veg and canonical not in input_vegs:
                        keep_kw = False
                        break

            if keep_kw:
                filtered_kws.append(kw)
        return ", ".join(filtered_kws)

    flavors: set[str] = set()
    if input_entities and isinstance(input_entities, dict):
        flavors.update(x.lower() for x in input_entities.get("flavor", set()) if x)
    cat_ents = winner_row.get("entities")
    if isinstance(cat_ents, dict):
        flavors.update(x.lower() for x in cat_ents.get("flavor", set()) if x)

    if flavors:
        kws = [k.strip() for k in str(match_gk).split(",") if k.strip()]
        return ", ".join(kw for kw in kws if not any(f in kw.lower() for f in flavors))

    return str(match_gk)


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
        reranker=None,
    ):
        self.ner = ner_engine
        self.embedder = embed_engine
        self.reranker = reranker if reranker is not None else embed_engine
        self.rules = logic_gates
        self.domain = domain
        self.classifier = classifier
        self.brands_df = brands_df

        self.flavor_categories = {}
        self.mixed_terms = {"mixed", "mix"}
        self.seafood_terms = {"seafood", "sea food"}
        self.veg_terms = {"vegetable", "vegetables", "veg", "veggie", "vege", "vegetarian"}

        if self.domain == config.DOMAIN_FOOD and brands_df is not None and not brands_df.empty:
            _, _, _, _, self.flavor_categories = build_food_flavors_info(brands_df)

            def get_all_terms_for(name: str) -> set[str]:
                terms = {name.lower()}
                row = brands_df[brands_df["Flavor Name"].str.lower() == name.lower()]
                if not row.empty:
                    aliases_str = str(row.iloc[0].get("Aliases", ""))
                    if aliases_str and aliases_str.lower() not in ("none", "nan"):
                        terms.update(x.strip().lower() for x in aliases_str.split(",") if x.strip())
                pluralized = []
                for t in terms:
                    if t.endswith("y"):
                        pluralized.append(t[:-1] + "ies")
                    elif not t.endswith("s"):
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

        self.exact_match_map: dict[str, list[int]] = {}
        self.token_sorted_map: dict[str, list[int]] = {}

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
                clean_txt = normalize_sku_text(standardize_sku_units(str(raw_name)))
            no_weights = str(raw_no_w or strip_weight_tokens(clean_txt)).strip()

            if clean_txt and clean_txt != "None":
                self.exact_match_map.setdefault(clean_txt, []).append(i)

            sorted_tokens = " ".join(sorted(no_weights.split())).strip()
            if sorted_tokens and sorted_tokens != "None":
                self.token_sorted_map.setdefault(sorted_tokens, []).append(i)

    def search_candidates(
        self,
        input_vec_dense: np.ndarray,
        input_vec_sparse: dict[str, float] | None = None,
        bt_filter: str | None = None,
        input_vec_bm25: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Retrieves top-K candidates from the vector store using hybrid RRF search."""
        hits = self.vector_store.search_catalog_hybrid(
            dense_vec=input_vec_dense,
            sparse_vec_dict=input_vec_sparse,
            domain=self.domain,
            top_k=config.TOP_K_RETRIEVAL,
            bt_filter=bt_filter,
            bm25_vec_dict=input_vec_bm25,
        )
        return pd.DataFrame(hits) if hits else pd.DataFrame()

    def _evaluate_candidate_pool(
        self,
        scored_cands: list[tuple[float, dict[str, Any]]],
        gate_limit: int,
        clean_input: str,
        input_entities: dict[str, set[str]],
        input_price: float,
        input_w_data: tuple,
        input_no_weights: str,
        input_description: str,
        input_category: str,
        predicted_bt: str | None,
    ) -> tuple[dict[str, Any] | None, float, str, str]:
        """Applies logic gates across top scored candidates and returns the winning candidate."""
        best_row = None
        best_score = -10.0
        best_status = "Rejected"
        best_reasons = ""
        best_raw_score = -100.0

        for cand_score, cand_row in scored_cands[:gate_limit]:
            gated_score, gated_status, gated_reasons = self.rules.apply_logic_gates(
                clean_input,
                input_entities,
                cand_row,
                cand_score,
                input_price,
                input_w_data,
                input_no_weights,
                domain=self.domain,
                input_description=input_description,
                input_category=input_category,
                predicted_bt=predicted_bt,
            )
            cand_clean = str(cand_row.get("clean_text", cand_row.get("Name", "")))
            best_clean = (
                str(best_row.get("clean_text", best_row.get("Name", "")))
                if best_row is not None
                else None
            )
            if is_better_gated_candidate(
                gated_score,
                cand_score,
                cand_clean,
                best_score if best_row is not None else None,
                best_raw_score if best_row is not None else None,
                best_clean,
            ):
                best_row = cand_row
                best_score = gated_score
                best_status = gated_status
                best_reasons = gated_reasons
                best_raw_score = cand_score

        return best_row, best_score, best_status, best_reasons

    def process_inputs(
        self,
        input_df: pd.DataFrame,
        progress_callback: Callable | None = None,
        chunk_size: int | None = None,
    ) -> pd.DataFrame:
        """Matches input SKUs against the catalog with automatic chunking."""
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

            clean_input = normalize_sku_text(standardize_sku_units(name))
            clean_inputs.append(clean_input)
            input_no_weights_list.append(strip_weight_tokens(clean_input))
            input_w_data_list.append(extract_weight_feature(clean_input))

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
                    consensus_indices = resolve_bt_consensus_indices(
                        cat_records, cat_indices, cat_bts_lower=cat_bts_lower
                    )
                    bypass_results[i] = (
                        cat_records[consensus_indices[0]],
                        1.0,
                        "High Confidence",
                        "Exact Text Match",
                    )
                else:
                    pending_exact[i] = cat_indices
                    needs_bt_or_ai_indices.append(i)
                continue

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
                    consensus_cands = resolve_bt_consensus_indices(
                        cat_records, cand_indices, cat_bts_lower=cat_bts_lower
                    )
                    selected_idx, weight_reason = resolve_weight_bypass_candidate(
                        cat_records, consensus_cands, input_w_data
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
            consensus_indices = resolve_bt_consensus_indices(
                cat_records, cat_indices, conf_bt=conf_bt, cat_bts_lower=cat_bts_lower
            )
            selected_idx = consensus_indices[0]
            bypass_results[i] = (
                cat_records[selected_idx],
                1.0,
                "High Confidence",
                "Exact Text Match",
            )

        for i, cand_indices in pending_fuzzy.items():
            conf_bt = bt_filters[i]
            valid_cands = resolve_bt_consensus_indices(
                cat_records, cand_indices, conf_bt=conf_bt, cat_bts_lower=cat_bts_lower
            )
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

        ai_entities: dict[int, dict[str, set[str]]] = {}
        all_ner_texts = [
            prepare_ner_text(build_ner_input_text(raw_names[i], descriptions[i], categories[i]))
            for i in range(len(raw_names))
        ]

        if self.ner and all_ner_texts:
            extracted_entities = self.ner.batch_extract_entities(all_ner_texts)
            for i, ents in enumerate(extracted_entities):
                ai_entities[i] = ents
        else:
            for i in range(len(all_ner_texts)):
                ai_entities[i] = {}

        if progress_callback:
            progress_callback(30.0)

        logger.info(f"[MATCH] Finalizing semantic retrieval matches for {len(raw_names)} items...")

        search_dense = []
        search_bm25 = []
        search_sparse = []
        search_filters = []
        search_map = []

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

            if bt_filters[i]:
                search_dense.append(dense)
                search_bm25.append(bm25)
                search_sparse.append(sparse)
                search_filters.append(bt_filters[i])
                search_map.append((i, "filtered"))

            search_dense.append(dense)
            search_bm25.append(bm25)
            search_sparse.append(sparse)
            search_filters.append(None)
            search_map.append((i, "unfiltered"))

            if i in stripped_embs_dense:
                search_dense.append(stripped_embs_dense[i])
                search_bm25.append(stripped_embs_bm25.get(i, {}))
                search_sparse.append(stripped_embs_sparse.get(i))
                search_filters.append(None)
                search_map.append((i, "stripped"))

        sku_candidates: dict[int, dict[str, list[dict[str, Any]]]] = {
            i: {"filtered": [], "unfiltered": [], "stripped": []} for i in ai_indices
        }
        if search_dense:
            all_candidates = self.vector_store.search_catalog_hybrid_batch(
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

        cross_pairs = []
        unique_pair_idx: dict[tuple[str, str], int] = {}
        pair_map = []

        for i in ai_indices:
            clean_in_no_w = input_no_weights_list[i]
            in_ents = ai_entities.get(i) or {}
            in_bf_key = "flavor" if self.domain == config.DOMAIN_FOOD else "brand"
            in_brand_or_flavor = ", ".join(sorted(str(x) for x in in_ents.get(in_bf_key, []) if x))
            formatted_query = format_matcher_rerank_text(
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
                key=lambda c: (
                    -round(float(c.get("_qdrant_score_", 0.0) or 0.0), 6),
                    str(c.get("clean_text", "")),
                    format_catalog_row_for_reranker(c, domain=self.domain),
                    str(c.get("basictype", c.get("BasicType", ""))),
                    str(c.get("db_uid", "")),
                )
            )

            seen_keys = set()
            seen_formatted = set()
            rerank_limit = getattr(config, "MATCHER_RERANK_CANDIDATES", 18)
            for cand_row in all_cands_list:
                cand_key = cand_row.get("db_uid") or cand_row.get("clean_text", "")
                if cand_key in seen_keys:
                    continue
                formatted_cand = format_catalog_row_for_reranker(cand_row, domain=self.domain)
                if formatted_cand not in seen_formatted and len(seen_formatted) >= rerank_limit:
                    continue
                seen_formatted.add(formatted_cand)
                seen_keys.add(cand_key)
                pair_key = (formatted_query, formatted_cand)
                p_idx = unique_pair_idx.get(pair_key)
                if p_idx is None:
                    p_idx = len(cross_pairs)
                    unique_pair_idx[pair_key] = p_idx
                    cross_pairs.append([formatted_query, formatted_cand])
                pair_map.append((i, cand_key, p_idx))

        cross_scores_list = []
        if cross_pairs:

            def _rerank_cb(done: int, total: int) -> None:
                if progress_callback and total > 0:
                    progress_callback(
                        50.0 + (25.0 * (done / total)),
                        f"Reranking {done} of {total}...",
                    )

            try:
                cross_scores_list = self.reranker.score_cross_encoder(
                    cross_pairs, progress_callback=_rerank_cb
                )
            except TypeError:
                cross_scores_list = self.reranker.score_cross_encoder(cross_pairs)

        if progress_callback:
            progress_callback(75.0)

        sku_scored_cands: dict[int, dict[Any, float]] = {i: {} for i in ai_indices}
        for sku_idx, cand_key, p_idx in pair_map:
            sku_scored_cands[sku_idx][cand_key] = float(cross_scores_list[p_idx])

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
                    str(item[1].get("db_uid", "")),
                )
            )
            return scored

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

                if bt_filter:
                    candidates_filtered = score_candidates(sku_candidates[i]["filtered"], i)
                    if candidates_filtered:
                        best_f, f_score, f_status, f_reasons = self._evaluate_candidate_pool(
                            candidates_filtered,
                            gate_limit,
                            clean_input,
                            input_entities,
                            current_input_price,
                            input_w_data,
                            input_no_weights,
                            input_description,
                            input_category,
                            bt_filter,
                        )
                        if f_status == "High Confidence":
                            best_match_row = best_f
                            final_score = f_score
                            status = f_status
                            reasons = f"BT-Filtered ({bt_filter}) | {f_reasons}"
                            bt_used = True
                        else:
                            best_match_row_fallback = best_f
                            final_score_fallback = f_score
                            status_fallback = f_status
                            reasons_fallback = (
                                f"BT-Filtered ({bt_filter}) [Low Confidence Fallback] | {f_reasons}"
                            )

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
                        best_uf, uf_score, uf_status, uf_reasons = self._evaluate_candidate_pool(
                            candidates_unfiltered,
                            gate_limit,
                            clean_input,
                            input_entities,
                            current_input_price,
                            input_w_data,
                            input_no_weights,
                            input_description,
                            input_category,
                            bt_filter,
                        )

                        if best_match_row_fallback is not None and final_score_fallback > uf_score:
                            best_match_row = best_match_row_fallback
                            final_score = final_score_fallback
                            status = status_fallback
                            reasons = reasons_fallback
                        else:
                            best_match_row = best_uf
                            final_score = uf_score
                            status = uf_status
                            if bt_filter:
                                reasons = f"BT-Fallback (BT: {bt_filter}, score: {uf_score:.4f}) | {uf_reasons}"
                            else:
                                reasons = uf_reasons

            assert best_match_row is not None
            match_gk = filter_matched_generic_keywords(
                match_gk=str(best_match_row.get("Generic keywords", "")),
                domain=self.domain,
                clean_input=clean_input,
                input_entities=input_entities,
                winner_row=best_match_row,
                rules=self.rules,
                flavor_categories=getattr(self, "flavor_categories", None),
                seafood_pattern=getattr(self, "_seafood_pattern", None),
                mixed_pattern=getattr(self, "_mixed_pattern", None),
                veg_pattern=getattr(self, "_veg_pattern", None),
                specific_flavor_pattern=getattr(self, "_specific_flavor_pattern", None),
            )

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
