import pytest
from engine.core.resource_loader import get_classifier, get_pipeline, get_ner_engine
from engine.pipeline.audit_engine import run_sku_audit
from engine.pipeline.processor import process_request
from engine.classification.loader import PRIMARY_DISH_TYPES, is_conflicting_dish_tag


class TestAuditAlignment:
    @classmethod
    def setup_class(cls):
        # Warm up models for food domain
        get_classifier("food")
        get_pipeline("food")

    def test_culinary_protein_prioritization(self):
        ner_food = get_ner_engine("food")
        
        def compute_suffixes(flavor_set):
            seafood_set = getattr(ner_food, "seafood_flavors", set())
            meat_set = getattr(ner_food, "meat_flavors", set())
            veg_set = getattr(ner_food, "vegetable_flavors", set())
            has_seafood = any(f in seafood_set for f in flavor_set)
            land_meats = [
                f for f in flavor_set
                if f in meat_set and f not in seafood_set and f not in ("egg", "mixed")
            ]
            meat_count = len(land_meats)
            has_other_protein = meat_count > 0 or has_seafood
            has_egg = ("egg" in flavor_set) and ("egg" in meat_set)
            has_veg = any(f in veg_set for f in flavor_set)

            if has_egg and not has_other_protein:
                meat_count = 1
                has_meat = True
            else:
                has_meat = meat_count > 0

            suffixes = []
            if (has_meat and has_seafood) or meat_count >= 2:
                suffixes.append("mixed")
            elif not has_seafood and not has_meat and has_veg:
                suffixes.append("veg")
            return suffixes

        # 1. Devilled Fish Set Menu has seafood
        sku_texts = ["Devilled Fish Set Menu"]
        ner_res = ner_food.batch_extract_entities(sku_texts)
        flavor_set = ner_res[0].get("flavor", set())
        assert compute_suffixes(flavor_set) == [], "Should NOT add seafood suffix (handled by rules engine) and NEVER add veg"

        # 2. Chicken with 'Mix' in description (should NOT trigger 'mixed' suffix)
        flavor_set_mix = {"chicken", "mixed", "carrot", "tomato"}
        assert compute_suffixes(flavor_set_mix) == [], "Chicken with 'mixed' meta-flavor must NOT get 'mixed' suffix"

        # 3. Egg + Real Meat (egg should be deprioritized, real meat prioritized, no mixed)
        flavor_set_egg_meat = {"chicken", "egg", "carrot"}
        assert compute_suffixes(flavor_set_egg_meat) == [], "Egg + Meat must prioritize meat and NOT add 'mixed'"

        # 4. Egg + Veg (no other meat/seafood: egg must be kept, NOT tagged as veg)
        flavor_set_egg_veg = {"egg", "carrot", "leek"}
        assert compute_suffixes(flavor_set_egg_veg) == [], "Egg + Veg must keep egg and NEVER add 'veg'"

        # 5. Egg alone (keep egg, no suffix)
        flavor_set_egg_alone = {"egg"}
        assert compute_suffixes(flavor_set_egg_alone) == []

        # 6. Real meat + Real meat (2 land meats -> 'mixed')
        flavor_set_multi_meat = {"chicken", "beef"}
        assert compute_suffixes(flavor_set_multi_meat) == ["mixed"]

        # 7. Meat + Seafood -> 'mixed'
        flavor_set_meat_seafood = {"chicken", "prawn"}
        assert compute_suffixes(flavor_set_meat_seafood) == ["mixed"]

        # 8. Veg alone -> 'veg'
        flavor_set_veg_alone = {"carrot", "cabbage"}
        assert compute_suffixes(flavor_set_veg_alone) == ["veg"]

    def test_cross_dish_contamination_filter(self):
        # Chop Suey Rice must be marked as conflicting dish under Fried Rice
        assert is_conflicting_dish_tag("Chop Suey Rice", "Fried Rice") is True
        assert is_conflicting_dish_tag("Vegetable Chop Suey Rice", "Fried Rice") is True
        assert is_conflicting_dish_tag("Chicken Biriyani", "Fried Rice") is True
        
        # Valid sub-dishes of Fried Rice must be allowed
        assert is_conflicting_dish_tag("Devilled Fish Fried Rice", "Fried Rice") is False
        assert is_conflicting_dish_tag("Fish Fried Rice", "Fried Rice") is False
        assert is_conflicting_dish_tag("Seafood Fried Rice", "Fried Rice") is False
        assert is_conflicting_dish_tag("Set Menu", "Fried Rice") is False
        assert is_conflicting_dish_tag("Seafood", "Fried Rice") is False

    def test_classifier_parity_audit_vs_processor(self):
        sku = {
            "name": "Devilled Fish Set Menu",
            "description": "",
            "category": "",
            "price": 1500.0,
        }
        
        proc_clf = process_request(task="classifier", domain="food", skus=[sku])["results"][0]
        audit_clf = run_sku_audit(
            sku_name=sku["name"],
            domain="food",
            task="classifier",
            price=sku["price"],
            description=sku["description"],
            category=sku["category"],
        )["final_output"]
        
        assert proc_clf["suggested_bt"] == audit_clf["suggested_bt"]
        assert proc_clf["suggested_gk"] == audit_clf["suggested_gk"]
        assert proc_clf["suggested_region"] == audit_clf["suggested_region"]
        
        # Ensure Chop Suey is NOT in suggested GK
        assert "Chop Suey" not in proc_clf["suggested_gk"]
        assert "Chop Suey" not in audit_clf["suggested_gk"]
        
        # Ensure rule 1 applied
        assert "Seafood" in proc_clf["suggested_gk"]
        assert "Seafood" in audit_clf["suggested_gk"]

    def test_pipeline_parity_audit_vs_processor(self):
        sku = {
            "name": "Devilled Fish Set Menu",
            "description": "",
            "category": "",
            "price": 1500.0,
        }
        
        proc_pip = process_request(task="pipeline", domain="food", skus=[sku])["results"][0]
        audit_pip = run_sku_audit(
            sku_name=sku["name"],
            domain="food",
            task="pipeline",
            price=sku["price"],
            description=sku["description"],
            category=sku["category"],
        )["final_output"]
        
        assert proc_pip["matched_catalog_name"] == audit_pip["matched_catalog_name"]
        assert proc_pip["score"] == audit_pip["score"]
        assert proc_pip["status"] == audit_pip["status"]
        assert proc_pip["suggested_bt"] == audit_pip["suggested_bt"]
        assert proc_pip["suggested_gk"] == audit_pip["suggested_gk"]
        assert proc_pip["suggested_region"] == audit_pip["suggested_region"]

    def test_devilled_chicken_burger_gk_preservation(self):
        sku = {
            "name": "Devilled Chicken Burger",
            "description": "",
            "category": "",
            "price": 1000.0,
        }
        proc_clf = process_request(task="classifier", domain="food", skus=[sku])["results"][0]
        audit_full = run_sku_audit(
            sku_name=sku["name"],
            domain="food",
            task="classifier",
            price=sku["price"],
            description=sku["description"],
            category=sku["category"],
        )
        audit_clf = audit_full["final_output"]

        # 1. Parity between processor and audit
        assert proc_clf["suggested_bt"] == audit_clf["suggested_bt"]
        assert proc_clf["suggested_gk"] == audit_clf["suggested_gk"]

        # 2. Both Chicken Burger and Spicy Chicken Burger are preserved in final GK
        assert "Chicken Burger" in proc_clf["suggested_gk"]
        assert "Spicy Chicken Burger" in proc_clf["suggested_gk"]
        assert "Burger" in proc_clf["suggested_gk"]

        # 3. Template enrichment correctly skipped because classifier BT was AUTO
        stage7 = audit_full.get("stage7_template_enrichment", {})
        assert stage7.get("matched_template") is False
        assert "AUTO" in stage7.get("note", "")

    def test_basictype_arcface_exact_match_guard(self):
        sku = {
            "name": "Chicken Burger",
            "description": "",
            "category": "",
            "price": 1300.0,
        }
        proc_res = process_request(task="matcher", domain="food", skus=[sku])["results"][0]
        audit_full = run_sku_audit(
            sku_name=sku["name"],
            domain="food",
            task="matcher",
            price=sku["price"],
            description=sku["description"],
            category=sku["category"],
        )
        audit_res = audit_full["final_output"]

        # 1. Exact match correctly resolves to Burger (not Mini Burger)
        assert proc_res["suggested_bt"] == "Burger"
        assert audit_res["suggested_bt"] == "Burger"
        assert proc_res["status"] == "High Confidence"
        assert audit_res["status"] == "High Confidence"
        assert proc_res["score"] == audit_res["score"]

    def test_chicken_noodles_mix_description_resolution(self):
        sku = {
            "name": "Chicken Noodles (Normal)",
            "description": "Carrot, Leaks, Eggs, Boiled Chicken Mix With Noodles. Served With Chili Paste Chicken Gravy & Tomato Sauce",
            "category": "NOODLES",
            "price": 1280.0,
        }
        proc_res = process_request(skus=[sku], task="pipeline", domain="food")["results"][0]
        audit_full = run_sku_audit(
            sku_name=sku["name"],
            domain="food",
            task="pipeline",
            price=sku["price"],
            description=sku["description"],
            category=sku["category"],
        )
        audit_res = audit_full["final_output"]

        # Ensure neither processor nor audit appended 'mixed' or matched to Mixed Noodles
        assert "mixed" not in proc_res.get("sku_name", "").lower()
        assert proc_res["matched_catalog_name"] == audit_res["matched_catalog_name"]
        assert proc_res["suggested_bt"] == "Fried Noodles"
        assert "Chicken Fried Noodles" in proc_res["suggested_gk"]
        assert proc_res["status"] == audit_res["status"]
        assert proc_res["status"] in ("High Confidence", "Medium Confidence")
