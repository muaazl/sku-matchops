import functools
import re

import numpy as np

from engine import config
from engine.config import (
    ALPHA,
    AUTO_THRESHOLD,
    FUSION_METHOD,
    REVIEW_THRESHOLD,
    RRF_K,
    TAG_SEARCH_LIMIT,
    USE_RERANKER,
)
from engine.config import (
    TOP_K_RETRIEVAL as TOP_K_FUSED,
)
from engine.nlp.text_cleaner import TextPipeline


def get_status(confidence: float, has_tags: bool, source: str = "") -> str:
    """Resolves classification status (AUTO, REVIEW, LOW) based on confidence and provenance."""
    if not has_tags:
        return "LOW"
    if source in ("override", "keyword"):
        return "AUTO"
    if source == "reranker":
        auto_thr = config.CONFIDENCE_THRESHOLD_HIGH
        review_thr = config.RERANKER_THRESHOLD
    else:
        auto_thr = AUTO_THRESHOLD
        review_thr = REVIEW_THRESHOLD
    if confidence >= auto_thr:
        return "AUTO"
    if confidence >= review_thr:
        return "REVIEW"
    return "LOW"


def resolve_gk_confidence_and_status(
    trained_conf: float,
    trained_source: str,
    final_conf: float,
    used_reranker: bool,
    has_tags: bool,
) -> tuple[float, str, str]:
    """Selects calibrated GK confidence, status, and winning source across trained vs reranked paths."""
    if trained_source == "trained" and trained_conf >= AUTO_THRESHOLD:
        gk_conf = float(trained_conf)
        gk_source = "trained"
    elif used_reranker:
        gk_conf = float(final_conf)
        gk_source = "reranker"
    else:
        gk_conf = float(final_conf)
        gk_source = ""
    return gk_conf, get_status(gk_conf, has_tags, gk_source), gk_source


def compute_weighted_confidence(
    bt_conf: float | None = None,
    gk_conf: float | None = None,
    region_conf: float | None = None,
    has_bt: bool = True,
    has_gk: bool = True,
    has_region: bool = True,
    w_bt: float = 0.5,
    w_gk: float = 0.3,
    w_reg: float = 0.2,
) -> float:
    """
    Computes dynamically normalized weighted confidence across available tags.
    Default weighting: 50% Basic Type, 30% Generic Keyword, 20% Region/Category.
    """
    weights = []
    confs = []
    if has_bt and bt_conf is not None:
        try:
            confs.append(float(bt_conf))
            weights.append(w_bt)
        except (ValueError, TypeError):
            pass
    if has_gk and gk_conf is not None:
        try:
            confs.append(float(gk_conf))
            weights.append(w_gk)
        except (ValueError, TypeError):
            pass
    if has_region and region_conf is not None:
        try:
            confs.append(float(region_conf))
            weights.append(w_reg)
        except (ValueError, TypeError):
            pass

    if not confs or sum(weights) == 0:
        try:
            return float(bt_conf or 0.0)
        except (ValueError, TypeError):
            return 0.0

    total_w = sum(weights)
    return round(sum(c * (w / total_w) for c, w in zip(confs, weights)), 4)


def _rrf_fusion(
    dense_results,
    bm25_results,
    sparse_results=None,
    k=RRF_K,
) -> list:
    """Reciprocal Rank Fusion across Dense, BM25, and optional BGE-M3 learned sparse hits."""
    scores = {}
    for rank_list in (dense_results, bm25_results, sparse_results):
        if not rank_list:
            continue
        for rank, hit in enumerate(rank_list):
            tag = hit.payload.get("tag")
            scores[tag] = scores.get(tag, 0.0) + 1.0 / (k + rank + 1)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"tag": tag, "score": score, "source": "rrf"} for tag, score in fused]


@functools.lru_cache(maxsize=128)
def _get_gk_regex(gk_tuple: tuple[str, ...]) -> re.Pattern:
    """Pre-compiles a single regex pattern for a list of GKs, optimized with alternation."""
    # Sort by length descending to match longest phrases first (e.g., 'bubble tea' before 'tea')
    sorted_gks = sorted(gk_tuple, key=len, reverse=True)
    escaped_gks = [re.escape(gk) for gk in sorted_gks]
    pattern_str = r"(?<![a-z0-9])(" + "|".join(escaped_gks) + r")(?![a-z0-9])"
    return re.compile(pattern_str)


def _weighted_fusion(dense_results, bm25_results, sparse_results=None, alpha=ALPHA) -> list:
    """Weighted Sum Fusion across Dense, BM25, and optional BGE-M3 learned sparse hits."""
    scores = {}

    for hit in dense_results or ():
        tag = hit.payload.get("tag")
        scores[tag] = scores.get(tag, 0) + alpha * hit.score

    lexical_lists = [lst for lst in (bm25_results, sparse_results) if lst]
    if lexical_lists:
        lex_weight = (1 - alpha) / len(lexical_lists)
        for lst in lexical_lists:
            for hit in lst:
                tag = hit.payload.get("tag")
                scores[tag] = scores.get(tag, 0) + lex_weight * hit.score

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"tag": tag, "score": score, "source": "weighted"} for tag, score in fused]


_TAGGER_FLAVOR_PATTERN_CACHE = {}


def _resolve_flavors_from_text(text: str, flavors_dict: dict) -> set:
    """Returns canonical flavor names found in a tag/BT/GK text string."""
    if not text or not flavors_dict:
        return set()
    dict_id = id(flavors_dict)
    pattern = _TAGGER_FLAVOR_PATTERN_CACHE.get(dict_id)
    if pattern is None:
        sorted_terms = sorted(flavors_dict.keys(), key=len, reverse=True)
        if sorted_terms:
            pattern = re.compile(
                r"(?<![a-z0-9])(" + "|".join(re.escape(t) for t in sorted_terms) + r")(?![a-z0-9])"
            )
        _TAGGER_FLAVOR_PATTERN_CACHE[dict_id] = pattern

    text_lower = text.lower()
    result = set()
    if pattern:
        for match in pattern.finditer(text_lower):
            canonical = flavors_dict.get(match.group(1))
            if canonical:
                result.add(canonical)
    return result


class DomainStrategy:
    @staticmethod
    def filter_search_tags(
        allowed_gks_for_bt: list, trained_conf: float, source: str, guaranteed: list
    ) -> tuple:
        return allowed_gks_for_bt, None  # search_allowed_tags, early_return

    @staticmethod
    def apply_post_search_filters(top_candidates: list, allowed_gks_lower: set) -> list:
        return top_candidates

    @staticmethod
    def inject_synthetic_tags(
        top_candidates: list, market_brand: str, bt: str, category: str, extracted_flavors: list
    ) -> list:
        return top_candidates

    @staticmethod
    def apply_flavor_leak_filters(
        guaranteed: list,
        trained_gk: list,
        top_candidates: list,
        extracted_flavors: list,
        ner_engine,
    ) -> tuple:
        return guaranteed, trained_gk, top_candidates

    @staticmethod
    def get_reranker_threshold(score: float, tag_lower: str, allowed_gks_lower: set) -> bool:
        return False

    @staticmethod
    def filter_final_trained_gk(trained_gk: list, allowed_gks_lower: set) -> list:
        return trained_gk


class FoodStrategy(DomainStrategy):
    @staticmethod
    def filter_search_tags(
        allowed_gks_for_bt: list, trained_conf: float, source: str, guaranteed: list
    ) -> tuple:
        if not allowed_gks_for_bt:
            # Prevent cross-contamination if no schema maps
            return None, (
                guaranteed,
                trained_conf if (source == "trained" and trained_conf >= AUTO_THRESHOLD) else 0.0,
            )
        return allowed_gks_for_bt, None

    @staticmethod
    def apply_post_search_filters(top_candidates: list, allowed_gks_lower: set) -> list:
        if allowed_gks_lower:
            return [c for c in top_candidates if c["tag"].lower().strip() in allowed_gks_lower]
        return top_candidates

    @staticmethod
    def inject_synthetic_tags(
        top_candidates: list, market_brand: str, bt: str, category: str, extracted_flavors: list
    ) -> list:
        if extracted_flavors:
            synthetic_tags = list(extracted_flavors)
            if len(extracted_flavors) > 1:
                synthetic_tags.append(" & ".join(extracted_flavors))

            existing_tags_lower = {c["tag"].lower().strip() for c in top_candidates}
            for st in synthetic_tags:
                if st.lower().strip() not in existing_tags_lower:
                    top_candidates.insert(
                        0, {"tag": st.title(), "score": 1.0, "source": "synthetic"}
                    )
        return top_candidates

    @staticmethod
    def apply_flavor_leak_filters(
        guaranteed: list,
        trained_gk: list,
        top_candidates: list,
        extracted_flavors: list,
        ner_engine,
    ) -> tuple:
        if ner_engine:
            input_flavors_canonical = {f.lower().strip() for f in (extracted_flavors or [])}

            def has_unrelated_flavor(tag: str) -> bool:
                gk_strong, gk_weak = ner_engine._get_dict_entities(tag)
                gk_flavors = gk_strong | gk_weak
                if gk_flavors and not gk_flavors.issubset(input_flavors_canonical):
                    return True
                return False

            guaranteed = [tag for tag in guaranteed if not has_unrelated_flavor(tag)]
            trained_gk = [tag for tag in trained_gk if not has_unrelated_flavor(tag)]
            top_candidates = [c for c in top_candidates if not has_unrelated_flavor(c["tag"])]
        return guaranteed, trained_gk, top_candidates

    @staticmethod
    def get_reranker_threshold(score: float, tag_lower: str, allowed_gks_lower: set) -> bool:
        return tag_lower in allowed_gks_lower and score >= config.RERANKER_THRESHOLD

    @staticmethod
    def filter_final_trained_gk(trained_gk: list, allowed_gks_lower: set) -> list:
        if allowed_gks_lower:
            return [t for t in trained_gk if t.lower().strip() in allowed_gks_lower]
        return trained_gk


class MarketStrategy(DomainStrategy):
    @staticmethod
    def filter_search_tags(
        allowed_gks_for_bt: list, trained_conf: float, source: str, guaranteed: list
    ) -> tuple:
        if allowed_gks_for_bt:
            return allowed_gks_for_bt, None
        return None, None

    @staticmethod
    def apply_post_search_filters(top_candidates: list, allowed_gks_lower: set) -> list:
        import engine.config as config

        if allowed_gks_lower:
            if getattr(config, "ALLOW_UNREGISTERED_TEMPLATE_KEYWORDS", True):
                return [
                    c
                    for c in top_candidates
                    if c["tag"].lower().strip() in allowed_gks_lower
                    or c.get("source") == "synthetic"
                ]
            else:
                return [c for c in top_candidates if c["tag"].lower().strip() in allowed_gks_lower]
        return top_candidates

    @staticmethod
    def inject_synthetic_tags(
        top_candidates: list, market_brand: str, bt: str, category: str, extracted_flavors: list
    ) -> list:
        if market_brand:
            synthetic_tags = []
            if bt:
                synthetic_tags.append(f"{market_brand} {bt}")
            if category:
                synthetic_tags.append(f"{market_brand} {category}")

            existing_tags_lower = {c["tag"].lower().strip() for c in top_candidates}
            for st in synthetic_tags:
                if st.lower().strip() not in existing_tags_lower:
                    top_candidates.insert(0, {"tag": st, "score": 1.0, "source": "synthetic"})
        return top_candidates

    @staticmethod
    def get_reranker_threshold(score: float, tag_lower: str, allowed_gks_lower: set) -> bool:
        if allowed_gks_lower:
            return tag_lower in allowed_gks_lower and score >= config.RERANKER_THRESHOLD
        return score >= config.RERANKER_THRESHOLD

    @staticmethod
    def filter_final_trained_gk(trained_gk: list, allowed_gks_lower: set) -> list:
        if allowed_gks_lower:
            return [t for t in trained_gk if t.lower().strip() in allowed_gks_lower]
        return trained_gk


def get_strategy(domain: str) -> DomainStrategy:
    if domain == "food":
        return FoodStrategy()
    elif domain == "market":
        return MarketStrategy()
    return DomainStrategy()


def tag_all_skus(
    sku_names,
    sku_categories,
    query_embeddings,
    vector_store,
    reranker,
    classifier,
    sku_descriptions=None,
    sku_prices=None,
    embed_engine=None,
    ner_engine=None,
    is_cancelled=None,
    progress_callback=None,
):
    n = len(sku_names)
    if n == 0:
        return []

    if sku_descriptions is None:
        sku_descriptions = [""] * n
    if sku_prices is None:
        sku_prices = [0.0] * n
    if sku_categories is None:
        sku_categories = [""] * n

    brands_per_sku = [""] * n
    flavors_per_sku = [[] for _ in range(n)]

    # 1. Batch NER extraction
    if ner_engine:
        combined_texts = []
        for idx_sku in range(n):
            name_txt = sku_names[idx_sku] or ""
            desc_txt = sku_descriptions[idx_sku] if idx_sku < len(sku_descriptions) else ""
            cat_txt = sku_categories[idx_sku] if idx_sku < len(sku_categories) else ""
            combined_texts.append(TextPipeline.build_ner_input(name_txt, desc_txt, cat_txt))

        if classifier.domain == "market":
            ner_results = ner_engine.batch_extract_entities(combined_texts)
            for i, ents in enumerate(ner_results):
                brand_set = ents.get("brand", set())
                if brand_set:
                    brands_per_sku[i] = sorted(brand_set, key=len, reverse=True)[0].title()
        elif classifier.domain == "food":
            ner_results = ner_engine.batch_extract_entities(combined_texts)
            for i, ents in enumerate(ner_results):
                flavor_set = ents.get("flavor", set())
                if flavor_set:
                    flavors_per_sku[i] = list(flavor_set)

    if is_cancelled and is_cancelled():
        raise InterruptedError("Job cancelled")
    if progress_callback:
        progress_callback(15.0)

    # Extract dense, BM25, and optional BGE-M3 learned sparse vectors
    if isinstance(query_embeddings, dict):
        dense_vecs = np.asarray(query_embeddings["dense"])
        bm25_vecs = query_embeddings.get("bm25") or [{} for _ in range(len(dense_vecs))]
        sparse_vecs = query_embeddings.get("sparse") or []
    else:
        dense_vecs = np.array([q["dense"] for q in query_embeddings])
        bm25_vecs = [q.get("bm25", {}) for q in query_embeddings]
        raw_sp = [q.get("sparse") for q in query_embeddings]
        sparse_vecs = [s or {} for s in raw_sp] if any(s for s in raw_sp) else []

    # 2. Batch predict BT (using multi-tier cold-start to warm-start fallback)
    raw_bt_preds = classifier.batch_predict_bt(
        dense_vecs, sku_prices, sku_names=sku_names, sku_descriptions=sku_descriptions
    )
    bt_results = [("", 0.0, "")] * n
    router_gks = {}
    for i in range(n):
        bt_tag, confidence, source, gks = raw_bt_preds[i]
        if gks:
            router_gks[i] = gks
        if confidence < REVIEW_THRESHOLD:
            bt_results[i] = ("", confidence, source)
            continue
        # Flavor conflict check for Food domain
        extracted_flavors = flavors_per_sku[i]
        if extracted_flavors and bt_tag:
            bt_flavors = _resolve_flavors_from_text(bt_tag, classifier.food_flavors_dict)
            input_flavors = set()
            for f in extracted_flavors:
                input_flavors.update(_resolve_flavors_from_text(f, classifier.food_flavors_dict))
            if bt_flavors and input_flavors and bt_flavors.isdisjoint(input_flavors):
                bt_results[i] = ("", 0.0, "conflict")
                continue
        bt_results[i] = (bt_tag, confidence, source)

    predicted_bts = [b[0] for b in bt_results]

    if is_cancelled and is_cancelled():
        raise InterruptedError("Job cancelled")
    if progress_callback:
        progress_callback(30.0)

    # 3. Batch predict Third Tag
    raw_third_tag_preds = classifier.batch_predict_third_tag(
        dense_vecs, sku_names, sku_descriptions, predicted_bts, sku_prices
    )
    third_tag_results = [("", 0.0, "")] * n
    for i in range(n):
        tag, conf, source = raw_third_tag_preds[i]
        third_tag_results[i] = (tag, conf, source)

    # 4. Batch predict GK
    trained_gk_preds = classifier.batch_predict_gk(dense_vecs, sku_prices)

    if is_cancelled and is_cancelled():
        raise InterruptedError("Job cancelled")
    if progress_callback:
        progress_callback(45.0)

    # 5. Hybrid GK Search Setup
    strategy = get_strategy(classifier.domain)
    bt_gk_map = getattr(classifier, "bt_gk_map", {})

    all_guaranteed = []
    all_allowed_gks_lower = []
    all_search_allowed_tags = []
    needs_search_indices = []
    early_return_results = {}

    for i in range(n):
        bt = predicted_bts[i]
        guaranteed = classifier.get_guaranteed_gk(bt)
        if not isinstance(guaranteed, list):
            guaranteed = list(guaranteed)

        # Propagate Tier 2 ML-kNN Generic Keywords if present from few-shot neighbors
        if i in router_gks:
            for extra_gk in router_gks[i]:
                if extra_gk not in guaranteed:
                    guaranteed.append(extra_gk)

        trained_gk, trained_conf, trained_source = trained_gk_preds[i]
        allowed_gks_for_bt = bt_gk_map.get(bt, [])
        allowed_gks_lower = {kw.lower().strip() for kw in allowed_gks_for_bt}

        # BT-as-GK fallback
        if bt and bt not in guaranteed:
            bt_lower = bt.lower().strip()
            all_known_gks_lower: set = set()
            for gk_list in bt_gk_map.values():
                for gk in gk_list:
                    all_known_gks_lower.add(gk.lower().strip())
            if bt_lower in all_known_gks_lower:
                guaranteed.append(bt)

        # Literal text match from SKU Name
        sku_name_lower = (sku_names[i] or "").lower()
        if allowed_gks_for_bt:
            gk_pool = [gk.lower().strip() for gk in allowed_gks_for_bt if gk.strip()]
            gk_map = {gk.lower().strip(): gk for gk in allowed_gks_for_bt if gk.strip()}
        else:
            gk_map = {}
            for gk_list in bt_gk_map.values():
                for gk in gk_list:
                    clean = gk.lower().strip()
                    if clean and clean not in gk_map:
                        gk_map[clean] = gk
            gk_pool = list(gk_map.keys())

        if gk_pool:
            pattern = _get_gk_regex(tuple(gk_pool))
            for match in pattern.finditer(sku_name_lower):
                matched_gk_lower = match.group(1)
                original_gk = gk_map[matched_gk_lower]
                if original_gk not in guaranteed:
                    guaranteed.append(original_gk)

        search_allowed_tags, early_return = strategy.filter_search_tags(
            allowed_gks_for_bt, trained_conf, trained_source, guaranteed
        )
        all_guaranteed.append(guaranteed)
        all_allowed_gks_lower.append(allowed_gks_lower)
        all_search_allowed_tags.append(search_allowed_tags)

        if early_return is not None:
            early_return_results[i] = early_return
        else:
            needs_search_indices.append(i)

    # 6. Execute Batch Vector Store Search
    gk_final_results = {}
    for i, early_res in early_return_results.items():
        _, tr_conf, tr_src = trained_gk_preds[i]
        early_conf, early_status, _ = resolve_gk_confidence_and_status(
            tr_conf, tr_src, early_res[1], False, bool(early_res[0])
        )
        gk_final_results[i] = (early_res[0], early_conf, early_status)

    if needs_search_indices:
        search_dense = [dense_vecs[i] for i in needs_search_indices]
        search_bm25 = [bm25_vecs[i] if i < len(bm25_vecs) else {} for i in needs_search_indices]
        search_sparse = [sparse_vecs[i] for i in needs_search_indices] if sparse_vecs else None
        search_allowed = [all_search_allowed_tags[i] for i in needs_search_indices]

        batch_search_hits = vector_store.search_batch_hybrid_tags(
            dense_queries=search_dense,
            sparse_queries=search_sparse,
            filter_dict_type="gk",
            domain=classifier.domain,
            allowed_tags_list=search_allowed,
            limit=TAG_SEARCH_LIMIT,
            bm25_queries=search_bm25,
        )

        if is_cancelled and is_cancelled():
            raise InterruptedError("Job cancelled")
        if progress_callback:
            progress_callback(70.0)

        # 7. Post-Search Fusion & Candidate Assembly
        all_rerank_pairs = []
        pair_mapping = []  # (list_idx, candidate_idx)
        per_item_top_candidates = []

        for list_idx, orig_idx in enumerate(needs_search_indices):
            dense_hits, bm25_hits, sparse_hits = batch_search_hits[list_idx]
            if FUSION_METHOD == "rrf":
                fused = _rrf_fusion(dense_hits, bm25_hits, sparse_hits)
            else:
                fused = _weighted_fusion(dense_hits, bm25_hits, sparse_hits)
            top_candidates = fused[:TOP_K_FUSED]
            top_candidates = strategy.apply_post_search_filters(
                top_candidates, all_allowed_gks_lower[orig_idx]
            )
            top_candidates = strategy.inject_synthetic_tags(
                top_candidates,
                brands_per_sku[orig_idx],
                predicted_bts[orig_idx],
                sku_categories[orig_idx],
                flavors_per_sku[orig_idx],
            )
            ner_eng = getattr(classifier, "ner_engine", ner_engine)
            guaranteed, trained_gk, top_candidates = strategy.apply_flavor_leak_filters(
                all_guaranteed[orig_idx],
                trained_gk_preds[orig_idx][0],
                top_candidates,
                flavors_per_sku[orig_idx],
                ner_eng,
            )
            all_guaranteed[orig_idx] = guaranteed
            # Store modified trained_gk
            trained_gk_preds[orig_idx] = (
                trained_gk,
                trained_gk_preds[orig_idx][1],
                trained_gk_preds[orig_idx][2],
            )
            per_item_top_candidates.append(top_candidates)

            if top_candidates and USE_RERANKER and reranker is not None:
                query_text = (sku_names[orig_idx] + " " + sku_descriptions[orig_idx]).strip()
                for c_idx, c in enumerate(top_candidates):
                    all_rerank_pairs.append([query_text, c["tag"]])
                    pair_mapping.append((list_idx, c_idx))

        # 8. Batch Cross-Encoder Reranking
        scored_candidates_per_item = {list_idx: [] for list_idx in range(len(needs_search_indices))}
        if all_rerank_pairs:
            raw_scores = reranker.predict(all_rerank_pairs)
            if isinstance(raw_scores, float) or (
                hasattr(raw_scores, "item") and raw_scores.ndim == 0
            ):
                scores = [float(raw_scores)]
            else:
                scores = [float(s) for s in raw_scores]

            for pair_idx, (list_idx, c_idx) in enumerate(pair_mapping):
                orig_idx = needs_search_indices[list_idx]
                c = per_item_top_candidates[list_idx][c_idx]
                score = scores[pair_idx]
                tag_lower = c["tag"].lower().strip()
                if strategy.get_reranker_threshold(
                    score, tag_lower, all_allowed_gks_lower[orig_idx]
                ):
                    scored_candidates_per_item[list_idx].append((c["tag"], score))

        if is_cancelled and is_cancelled():
            raise InterruptedError("Job cancelled")
        if progress_callback:
            progress_callback(90.0)

        # 9. Merge Final Tags for searched items
        for list_idx, orig_idx in enumerate(needs_search_indices):
            top_candidates = per_item_top_candidates[list_idx]
            scored_candidates = scored_candidates_per_item[list_idx]
            scored_candidates.sort(key=lambda x: x[1], reverse=True)

            used_reranker = bool(USE_RERANKER and reranker is not None and top_candidates)
            if used_reranker:
                if scored_candidates:
                    top_score = scored_candidates[0][1]
                    margin_cutoff = max(
                        config.RERANKER_THRESHOLD, top_score - config.RERANKER_MARGIN
                    )
                    scored_candidates = [c for c in scored_candidates if c[1] >= margin_cutoff]
                reranked_tags = [tag for tag, score in scored_candidates]
                final_conf = scored_candidates[0][1] if scored_candidates else 0.0
            else:
                reranked_tags = [c["tag"] for c in top_candidates]
                final_conf = min(1.0, top_candidates[0]["score"]) if top_candidates else 0.0

            seen = set()
            merged = []
            trained_gk, trained_conf, trained_source = trained_gk_preds[orig_idx]
            trained_gk = strategy.filter_final_trained_gk(
                trained_gk, all_allowed_gks_lower[orig_idx]
            )

            for tag in all_guaranteed[orig_idx] + trained_gk + reranked_tags:
                key = tag.lower().strip()
                if key not in seen:
                    seen.add(key)
                    merged.append(tag)

            gk_conf, gk_status, _ = resolve_gk_confidence_and_status(
                trained_conf, trained_source, final_conf, used_reranker, bool(merged)
            )
            gk_final_results[orig_idx] = (merged, gk_conf, gk_status)

    # 10. Assemble Final Output Dicts
    results = []
    domain = classifier.domain
    active_model = getattr(classifier, "active_bt_model", "logreg")
    tag_key = "suggested_region" if domain == "food" else "suggested_category"
    conf_key = "region_confidence" if domain == "food" else "category_confidence"
    status_key = "region_status" if domain == "food" else "category_status"
    source_key = "region_source" if domain == "food" else "category_source"

    for i in range(n):
        bt_tag, bt_conf, bt_source = bt_results[i]
        gk_tags, gk_conf, gk_status = gk_final_results.get(i, ([], 0.0, "LOW"))
        third_tag_name, third_tag_conf, third_tag_source = third_tag_results[i]

        res = {
            "suggested_gk": ", ".join(gk_tags),
            "gk_confidence": round(gk_conf, 3),
            "gk_status": gk_status,
            "suggested_bt": bt_tag,
            "bt_confidence": round(bt_conf, 3),
            "bt_status": get_status(bt_conf, bool(bt_tag), bt_source),
            "bt_source": bt_source,
            "bt_model": active_model,
            "model": active_model,
            tag_key: third_tag_name,
            conf_key: round(third_tag_conf, 3),
            status_key: get_status(third_tag_conf, bool(third_tag_name), third_tag_source),
            source_key: third_tag_source,
        }
        gk_str = ", ".join(gk_tags)
        third_label = "Region" if domain == "food" else "Category"
        reasoning = (
            f"Classifier ({active_model}): BT='{bt_tag}' ({bt_source}, {round(bt_conf, 3)}), "
            f"GK='{gk_str}' ({round(gk_conf, 3)}), "
            f"{third_label}='{third_tag_name}' ({third_tag_source}, {round(third_tag_conf, 3)})"
        )
        res["reasoning"] = reasoning
        results.append(res)

    if progress_callback:
        progress_callback(100.0)

    return results
