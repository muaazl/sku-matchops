import functools
import logging
import re
from typing import Dict, List, Set, Tuple
import warnings

# Suppress Hugging Face FastTokenizer regex deprecation warnings
warnings.filterwarnings("ignore", message=".*fix_mistral_regex.*")
warnings.filterwarnings("ignore", message=".*The regex pattern.*")

import pandas as pd
from rapidfuzz import fuzz, process

from engine import config

logger = logging.getLogger("matchops.ner")

class NEREngine:
    """Named Entity Recognition engine (GLiNER ONNX-optimised).

    Extraction strategy (food domain):
      Layer 1 – Flavor alias dict  (Food_Flavors Google Sheet)
                Covers all flavors: proteins (chicken, lamb), sweets (chocolate, vanilla), etc.
      Layer 2 – GLiNER NER fallback, label = "flavor"
                Only runs when the dict finds nothing — zero extra cost otherwise.

    Market domain uses the same 2-layer approach for "brand".

    NOTE: entities["protein"] does NOT exist. Everything lives under entities["flavor"].
    The dynamically loaded food_flavors_dict is used by the conflict-check logic to identify
    flavor conflicts across SKUs.
    """

    def __init__(self, brands_df: pd.DataFrame, domain: str = config.DOMAIN_MARKET, shared_model=None):
        self.domain = domain
        self.target_label = "flavor" if domain == config.DOMAIN_FOOD else "brand"
        self.labels = config.NER_LABELS
        self.model = shared_model
        if self.model is None:
            self._load_model()

        self.brand_mapping: Dict[str, str] = {}
        self.weak_brands: Set[str] = set()
        self.meat_flavors: Set[str] = set()
        self.vegetable_flavors: Set[str] = set()
        self.seafood_flavors: Set[str] = set()
        self.sorted_aliases: List[str] = []
        self.combined_pattern = None
        self._build_brand_knowledge(brands_df)

    # ─────────────────────────────────────────────────────────────
    # Model Loading
    # ─────────────────────────────────────────────────────────────

    def _load_model(self):
        logger.info("[NER] Loading GLiNER from local ONNX directory...")
        from gliner import GLiNER
        try:
            # Load from the local directory (engine/onnx_models/gliner).
            # fix_mistral_regex=True silences the incorrect regex pattern warning from the tokenizer.
            # load_onnx_model=True uses the model.onnx file for faster CPU inference.
            self.model = GLiNER.from_pretrained(
                config.GLINER_MODEL,
                load_onnx_model=True,
                fix_mistral_regex=True,
            )
            logger.info("[NER] GLiNER loaded successfully in ONNX mode.")
        except Exception as e:
            logger.warning(f"[NER] ONNX load failed, falling back to PyTorch: {e}")
            try:
                self.model = GLiNER.from_pretrained(
                    config.GLINER_MODEL,
                    local_files_only=True,
                    fix_mistral_regex=True,
                )
                logger.info("[NER] GLiNER loaded successfully in PyTorch mode.")
            except Exception as e2:
                logger.error(f"[NER] All GLiNER load attempts failed: {e2}")
                self.model = None

    def warmup(self):
        if self.model:
            logger.info("[NER] GLiNER ready.")

    # ─────────────────────────────────────────────────────────────
    # Dictionary Building
    # ─────────────────────────────────────────────────────────────

    def _build_brand_knowledge(self, df: pd.DataFrame):
        """Indexes flavor/brand aliases from the Google Sheet for fast dict scanning."""
        if df is None or df.empty:
            return
        for _, row in df.iterrows():
            canonical = str(row.get("Brand Name", row.get("Flavor Name", ""))).strip().lower()
            if not canonical or canonical == "nan":
                continue
            aliases_str = str(row.get("Aliases", ""))
            aliases = (
                [x.strip().lower() for x in aliases_str.split(",")
                 if x.strip() and x.strip().lower() != "nan"]
                if aliases_str != "nan" else []
            )
            aliases.append(canonical)
            is_weak = str(row.get("Is_Weak", "")).strip().lower() in ("true", "1", "yes", "y")
            if is_weak:
                self.weak_brands.add(canonical)
                
            if str(row.get("Is_Meat", "")).strip().lower() in ("true", "1", "yes", "y"):
                self.meat_flavors.add(canonical)
            if str(row.get("Is_Vegetable", "")).strip().lower() in ("true", "1", "yes", "y"):
                self.vegetable_flavors.add(canonical)
            if str(row.get("Is_Seafood", "")).strip().lower() in ("true", "1", "yes", "y"):
                self.seafood_flavors.add(canonical)
                
            for alias in aliases:
                if alias and alias != "nan":
                    self.brand_mapping[alias] = canonical
        self.sorted_aliases = sorted(self.brand_mapping.keys(), key=len, reverse=True)
        if self.sorted_aliases:
            self.combined_pattern = re.compile(
                r"(?<![a-z0-9])(" + "|".join(re.escape(a) for a in self.sorted_aliases) + r")(?![a-z0-9])"
            )
        else:
            self.combined_pattern = None

    # ─────────────────────────────────────────────────────────────
    # Layer 1: Flavor / Brand Alias Dictionary
    # ─────────────────────────────────────────────────────────────

    @functools.lru_cache(maxsize=100000)
    def _get_dict_entities(self, text: str) -> Tuple[frozenset[str], frozenset[str]]:
        """Scans text against the alias dictionary using word-boundary regex + fuzzy fallback."""
        text_lower = text.lower()
        found_matches = []

        if self.combined_pattern:
            def replace_callback(match):
                alias = match.group(1)
                canonical = self.brand_mapping[alias]
                is_weak = canonical in self.weak_brands
                found_matches.append((canonical, match.start(), is_weak))
                return " "
            text_lower = self.combined_pattern.sub(replace_callback, text_lower)

        # Fuzzy fallback only when dict finds nothing
        if not found_matches and self.sorted_aliases:
            words = [w for w in re.findall(r"\b\w+\b", text_lower) if len(w) >= 4]
            bi_grams = [" ".join(words[i:i+2]) for i in range(len(words) - 1)]
            for candidate in words + bi_grams:
                m = process.extractOne(candidate, self.sorted_aliases, scorer=fuzz.ratio, score_cutoff=85)
                if m:
                    canonical = self.brand_mapping[m[0]]
                    is_weak = canonical in self.weak_brands
                    found_matches.append((canonical, 999999, is_weak))
                    break

        found_strong = set()
        found_weak = set()
        if found_matches:
            found_matches.sort(key=lambda x: x[1])
            if self.target_label == "brand":
                # Brands: first match only
                best = found_matches[0]
                (found_weak if best[2] else found_strong).add(best[0])
            else:
                # Flavors: all matches
                for canonical, _, is_weak in found_matches:
                    (found_weak if is_weak else found_strong).add(canonical)
        return frozenset(found_strong), frozenset(found_weak)

    # ─────────────────────────────────────────────────────────────
    # Layer 2: NER Fallback
    # ─────────────────────────────────────────────────────────────

    def _resolve_entity_text(self, entity_text: str) -> Set[str]:
        """Maps an extracted NER entity string to known canonical aliases or raw text."""
        value = entity_text.lower().strip()
        if not value or value in ("product", "product:"):
            return set()
        words = [w.strip() for w in re.findall(r"\b\w+\b", value) if w.strip()]
        matched = {self.brand_mapping[w] for w in words if w in self.brand_mapping}
        return matched if matched else {value}

    def _run_ner(self, text: str) -> List[Dict]:
        try:
            if self.model:
                return self.model.predict_entities(text, self.labels, threshold=0.25)
            logger.warning(f"[NER] No model available for '{text}'.")
            return []
        except Exception as e:
            logger.error(f"[NER] Prediction failed: {e}")
            return []

    # ─────────────────────────────────────────────────────────────
    # Public: Single Extract
    # ─────────────────────────────────────────────────────────────

    def extract_entities(self, text: str) -> Dict[str, Set[str]]:
        """2-layer extraction. Result dict contains only 'flavor' (food) or 'brand' (market)."""
        if not isinstance(text, str) or not text.strip():
            return {label: set() for label in self.labels}

        extracted: Dict[str, Set[str]] = {label: set() for label in self.labels}

        # Layer 1: dictionary
        dict_strong, dict_weak = self._get_dict_entities(text)
        extracted[self.target_label].update(dict_strong)
        if not extracted[self.target_label]:
            extracted[self.target_label].update(dict_weak)

        # Layer 2: NER fallback — only when dict finds nothing
        if not extracted[self.target_label]:
            for entity in self._run_ner(text):
                label = entity["label"]
                resolved = self._resolve_entity_text(entity["text"])
                extracted[label].update(resolved)

        return extracted

    # ─────────────────────────────────────────────────────────────
    # Public: Batch Extract
    # ─────────────────────────────────────────────────────────────

    def batch_extract_entities(
        self, texts: List[str], batch_size: int = config.EMBED_BATCH_SIZE
    ) -> List[Dict[str, Set[str]]]:
        """Batch extraction: dict first, NER only for texts where dict found nothing."""
        if not texts:
            return []

        all_results: List[Dict[str, Set[str]]] = []
        valid_texts = [t if (isinstance(t, str) and t.strip()) else "" for t in texts]
        fallback_indices: List[int] = []

        for i, text in enumerate(valid_texts):
            extracted: Dict[str, Set[str]] = {label: set() for label in self.labels}
            if text:
                dict_strong, dict_weak = self._get_dict_entities(text)
                extracted[self.target_label].update(dict_strong)
                if not extracted[self.target_label]:
                    extracted[self.target_label].update(dict_weak)
                if not extracted[self.target_label]:
                    fallback_indices.append(i)
            all_results.append(extracted)

        if not fallback_indices:
            return all_results

        # Batch NER for texts that got no dict hit
        fallback_texts = [valid_texts[i] for i in fallback_indices]
        try:
            if self.model:
                if hasattr(self.model, "inference"):
                    raw_list = self.model.inference(
                        fallback_texts, self.labels, threshold=0.25, batch_size=batch_size
                    )
                elif hasattr(self.model, "batch_predict_entities"):
                    raw_list = self.model.batch_predict_entities(
                        fallback_texts, self.labels, threshold=0.25, batch_size=batch_size
                    )
                else:
                    raw_list = [
                        self.model.predict_entities(t, self.labels, threshold=0.25) if t else []
                        for t in fallback_texts
                    ]
            else:
                logger.warning("[NER] No model for batch NER; returning empty for fallback items.")
                raw_list = [[] for _ in fallback_texts]

            for list_idx, original_idx in enumerate(fallback_indices):
                for entity in raw_list[list_idx]:
                    label = entity["label"]
                    resolved = self._resolve_entity_text(entity["text"])
                    all_results[original_idx][label].update(resolved)

        except Exception as e:
            logger.warning(f"[NER] Batch NER failed, falling back to sequential: {e}")
            for list_idx, original_idx in enumerate(fallback_indices):
                all_results[original_idx] = self.extract_entities(valid_texts[original_idx])

        return all_results
