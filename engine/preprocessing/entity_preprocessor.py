"""Layer-1 brand and flavor alias dictionary indexer and matcher."""

from __future__ import annotations

import functools
import re

import pandas as pd
from rapidfuzz import fuzz, process


class EntityAliasIndex:
    """Indexes canonical brand/flavor aliases and matches them via word-boundary regex + fuzzy fallback."""

    def __init__(self, brands_df: pd.DataFrame | None = None, target_label: str = "brand") -> None:
        self.target_label = target_label
        self.brand_mapping: dict[str, str] = {}
        self.weak_brands: set[str] = set()
        self.meat_flavors: set[str] = set()
        self.vegetable_flavors: set[str] = set()
        self.seafood_flavors: set[str] = set()
        self.sorted_aliases: list[str] = []
        self.combined_pattern: re.Pattern | None = None

        self.build_entity_alias_index(brands_df)
        # Per-instance memoization avoids pinning discarded instances in a class-level lru_cache.
        self.extract_dictionary_entities = functools.lru_cache(maxsize=100000)(
            self._extract_dictionary_entities_uncached
        )

    def build_entity_alias_index(self, df: pd.DataFrame | None) -> None:
        """Builds alias lookup maps, flavor category sets, and compiled regex from `df`."""
        self.brand_mapping = {}
        self.weak_brands = set()
        self.meat_flavors = set()
        self.vegetable_flavors = set()
        self.seafood_flavors = set()
        self.sorted_aliases = []
        self.combined_pattern = None

        if df is None or df.empty:
            return

        cols = list(df.columns)
        name_col = (
            "Brand Name"
            if "Brand Name" in cols
            else ("Flavor Name" if "Flavor Name" in cols else None)
        )
        names = df[name_col].tolist() if name_col else [""] * len(df)
        aliases_list = df["Aliases"].tolist() if "Aliases" in cols else [""] * len(df)
        is_weaks = df["Is_Weak"].tolist() if "Is_Weak" in cols else [""] * len(df)
        is_meats = df["Is_Meat"].tolist() if "Is_Meat" in cols else [""] * len(df)
        is_vegs = df["Is_Vegetable"].tolist() if "Is_Vegetable" in cols else [""] * len(df)
        is_seafoods = df["Is_Seafood"].tolist() if "Is_Seafood" in cols else [""] * len(df)

        for raw_name, raw_alias, raw_weak, raw_meat, raw_veg, raw_seafood in zip(
            names, aliases_list, is_weaks, is_meats, is_vegs, is_seafoods
        ):
            canonical = str(raw_name or "").strip().lower()
            if not canonical or canonical == "nan":
                continue
            aliases_str = str(raw_alias or "")
            aliases = (
                [
                    x.strip().lower()
                    for x in aliases_str.split(",")
                    if x.strip() and x.strip().lower() != "nan"
                ]
                if aliases_str != "nan"
                else []
            )
            aliases.append(canonical)
            if str(raw_weak or "").strip().lower() in ("true", "1", "yes", "y"):
                self.weak_brands.add(canonical)
            if str(raw_meat or "").strip().lower() in ("true", "1", "yes", "y"):
                self.meat_flavors.add(canonical)
            if str(raw_veg or "").strip().lower() in ("true", "1", "yes", "y"):
                self.vegetable_flavors.add(canonical)
            if str(raw_seafood or "").strip().lower() in ("true", "1", "yes", "y"):
                self.seafood_flavors.add(canonical)

            for alias in aliases:
                if alias and alias != "nan":
                    self.brand_mapping[alias] = canonical

        self.sorted_aliases = sorted(self.brand_mapping.keys(), key=len, reverse=True)
        if self.sorted_aliases:
            self.combined_pattern = re.compile(
                r"(?<![a-z0-9])("
                + "|".join(re.escape(a) for a in self.sorted_aliases)
                + r")(?![a-z0-9])"
            )

    def _extract_dictionary_entities_uncached(
        self, text: str
    ) -> tuple[frozenset[str], frozenset[str]]:
        """Scans text against the alias dictionary using word-boundary regex + fuzzy fallback."""
        text_lower = text.lower()
        found_matches: list[tuple[str, int, bool]] = []

        if self.combined_pattern:

            def _replace_callback(match: re.Match) -> str:
                canonical = self.brand_mapping[match.group(1)]
                found_matches.append((canonical, match.start(), canonical in self.weak_brands))
                return " "

            text_lower = self.combined_pattern.sub(_replace_callback, text_lower)

        if not found_matches and self.sorted_aliases:
            words = [w for w in re.findall(r"\b\w+\b", text_lower) if len(w) >= 4]
            bi_grams = [" ".join(words[i : i + 2]) for i in range(len(words) - 1)]
            for candidate in words + bi_grams:
                match = process.extractOne(
                    candidate, self.sorted_aliases, scorer=fuzz.ratio, score_cutoff=85
                )
                if match:
                    canonical = self.brand_mapping[match[0]]
                    found_matches.append((canonical, 999999, canonical in self.weak_brands))
                    break

        found_strong: set[str] = set()
        found_weak: set[str] = set()
        if found_matches:
            found_matches.sort(key=lambda x: x[1])
            if self.target_label == "brand":
                best = found_matches[0]
                (found_weak if best[2] else found_strong).add(best[0])
            else:
                for canonical, _, is_weak in found_matches:
                    (found_weak if is_weak else found_strong).add(canonical)
        return frozenset(found_strong), frozenset(found_weak)

    def resolve_entity_alias(self, entity_text: str) -> set[str]:
        """Maps an extracted entity string to known canonical aliases or retains raw text."""
        value = entity_text.lower().strip()
        if not value or value in ("product", "product:"):
            return set()
        words = [w.strip() for w in re.findall(r"\b\w+\b", value) if w.strip()]
        matched = {self.brand_mapping[w] for w in words if w in self.brand_mapping}
        return matched if matched else {value}
