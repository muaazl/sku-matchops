"""GLiNER ONNX named entity extractor with alias-dictionary fast path and LRU cache."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from pathlib import Path

import pandas as pd

from engine.core import config
from engine.core.observability import ONNX_INFERENCE_SECONDS
from engine.preprocessing.entity_preprocessor import EntityAliasIndex

logger = logging.getLogger("matchops.inference.gliner")


class GlinerModel:
    """Two-layer entity recognizer using alias dictionary matching plus GLiNER ONNX fallback."""

    def __init__(
        self, brands_df: pd.DataFrame, domain: str = config.DOMAIN_MARKET, shared_model=None
    ) -> None:
        self.domain = domain
        self.target_label = "flavor" if domain == config.DOMAIN_FOOD else "brand"
        self.labels = config.NER_LABELS
        self.model = shared_model
        if self.model is None:
            self.load_gliner_model()

        self._alias_index = EntityAliasIndex(brands_df, target_label=self.target_label)
        self.brand_mapping = self._alias_index.brand_mapping
        self.weak_brands = self._alias_index.weak_brands
        self.meat_flavors = self._alias_index.meat_flavors
        self.vegetable_flavors = self._alias_index.vegetable_flavors
        self.seafood_flavors = self._alias_index.seafood_flavors
        self.sorted_aliases = self._alias_index.sorted_aliases
        self.combined_pattern = self._alias_index.combined_pattern
        self._get_dict_entities = self._alias_index.extract_dictionary_entities

        self._ner_cache: OrderedDict[str, list[dict]] = OrderedDict()
        self._ner_cache_max = 50000
        self._cache_lock = threading.RLock()

    def load_gliner_model(self) -> None:
        """Loads the local GLiNER ONNX model strictly without network downloads."""
        from gliner import GLiNER

        onnx_path = Path(config.GLINER_ONNX)
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"[FAIL-FAST] Required GLiNER ONNX model missing at '{onnx_path}'. "
                "Run 'python -m scripts.sync --models sync --model-target gliner' first."
            )

        try:
            self.model = GLiNER.from_pretrained(
                str(config.GLINER_MODEL),
                load_onnx_model=True,
                onnx_model_file=onnx_path.name,
                local_files_only=True,
                fix_mistral_regex=True,
            )
            tok = getattr(
                getattr(self.model, "data_processor", None), "transformer_tokenizer", None
            )
            if tok is not None and getattr(tok, "model_max_length", 0) > 100000:
                tok.model_max_length = 512
            logger.info("[NER] GLiNER ONNX loaded strictly from %s.", onnx_path)
        except Exception as exc:
            raise RuntimeError(
                f"[FAIL-FAST] Failed to load GLiNER ONNX model from '{onnx_path}': {exc}"
            ) from exc

    def warmup(self) -> None:
        """Logs readiness when the GLiNER ONNX runtime model is initialized."""
        if self.model:
            logger.info("[NER] GLiNER ready.")

    def _resolve_entity_text(self, entity_text: str) -> set[str]:
        """Maps an extracted NER string to known canonical aliases or raw fallback text."""
        alias_index = getattr(self, "_alias_index", None)
        if alias_index is not None:
            return alias_index.resolve_entity_alias(entity_text)
        value = entity_text.lower().strip()
        return set() if not value or value in ("product", "product:") else {value}

    def predict_raw_entities(self, text: str) -> list[dict]:
        """Runs single-text GLiNER inference with thread-safe LRU memoization."""
        cache = getattr(self, "_ner_cache", None)
        lock = getattr(self, "_cache_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._cache_lock = lock
        with lock:
            if cache is not None and text in cache:
                cache.move_to_end(text)
                return cache[text]
        try:
            if self.model:
                res = self.model.predict_entities(
                    text, self.labels, threshold=config.NER_CONFIDENCE_THRESHOLD
                )
                with lock:
                    if cache is not None:
                        cache[text] = res
                        cache.move_to_end(text)
                        if len(cache) > getattr(self, "_ner_cache_max", 50000):
                            cache.popitem(last=False)
                return res
            logger.warning("[NER] No model available for '%s'.", text)
            return []
        except Exception as exc:
            logger.error("[NER] Prediction failed: %s", exc)
            return []

    def extract_entities(self, text: str) -> dict[str, set[str]]:
        """Extracts brand or flavor entities using dictionary lookup before GLiNER fallback."""
        if not isinstance(text, str) or not text.strip():
            return {label: set() for label in self.labels}

        extracted: dict[str, set[str]] = {label: set() for label in self.labels}

        dict_strong, dict_weak = self._get_dict_entities(text)
        extracted[self.target_label].update(dict_strong)
        if not extracted[self.target_label]:
            extracted[self.target_label].update(dict_weak)

        if not extracted[self.target_label]:
            for entity in self.predict_raw_entities(text):
                label = entity["label"]
                extracted[label].update(self._resolve_entity_text(entity["text"]))

        return extracted

    def batch_extract_entities(
        self, texts: list[str], batch_size: int = config.ONNX_BATCH_SIZE
    ) -> list[dict[str, set[str]]]:
        """Batch-extracts entities with dictionary fast path and batched GLiNER fallback."""
        if not texts:
            return []

        all_results: list[dict[str, set[str]]] = []
        valid_texts = [t if (isinstance(t, str) and t.strip()) else "" for t in texts]
        fallback_indices: list[int] = []

        for i, text in enumerate(valid_texts):
            extracted: dict[str, set[str]] = {label: set() for label in self.labels}
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

        cache = getattr(self, "_ner_cache", None)
        lock = getattr(self, "_cache_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._cache_lock = lock
        cache_max = getattr(self, "_ner_cache_max", 50000)
        resolved_raw: dict[str, list[dict]] = {}
        missing_texts: list[str] = []
        missing_seen: set[str] = set()

        with lock:
            for idx in fallback_indices:
                txt = valid_texts[idx]
                if cache is not None and txt in cache:
                    cache.move_to_end(txt)
                    resolved_raw[txt] = cache[txt]
                elif txt not in missing_seen:
                    missing_seen.add(txt)
                    missing_texts.append(txt)

        t_ner = time.perf_counter()
        try:
            if missing_texts:
                if self.model:
                    if hasattr(self.model, "inference"):
                        raw_list = self.model.inference(
                            missing_texts,
                            self.labels,
                            threshold=config.NER_CONFIDENCE_THRESHOLD,
                            batch_size=batch_size,
                        )
                    elif hasattr(self.model, "batch_predict_entities"):
                        raw_list = self.model.batch_predict_entities(
                            missing_texts,
                            self.labels,
                            threshold=config.NER_CONFIDENCE_THRESHOLD,
                            batch_size=batch_size,
                        )
                    else:
                        raw_list = [
                            self.model.predict_entities(
                                t, self.labels, threshold=config.NER_CONFIDENCE_THRESHOLD
                            )
                            if t
                            else []
                            for t in missing_texts
                        ]
                else:
                    logger.warning(
                        "[NER] No model for batch NER; returning empty for fallback items."
                    )
                    raw_list = [[] for _ in missing_texts]
                ONNX_INFERENCE_SECONDS.labels(model="ner").observe(time.perf_counter() - t_ner)

                with lock:
                    for txt, ents_raw in zip(missing_texts, raw_list):
                        resolved_raw[txt] = ents_raw
                        if cache is not None and self.model is not None:
                            cache[txt] = ents_raw
                            cache.move_to_end(txt)
                    if cache is not None:
                        while len(cache) > cache_max:
                            cache.popitem(last=False)

            for original_idx in fallback_indices:
                for entity in resolved_raw.get(valid_texts[original_idx], ()):
                    label = entity["label"]
                    all_results[original_idx][label].update(
                        self._resolve_entity_text(entity["text"])
                    )
        except Exception as exc:
            logger.warning("[NER] Batch NER failed, falling back to sequential: %s", exc)
            for original_idx in fallback_indices:
                all_results[original_idx] = self.extract_entities(valid_texts[original_idx])

        return all_results
