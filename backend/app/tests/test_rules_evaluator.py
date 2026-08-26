import unittest
from unittest.mock import MagicMock

from engine.rules_engine.evaluator import evaluate_conditions, _evaluate_single_condition
from engine.rules_engine.loader import Rule

class TestRulesEvaluator(unittest.TestCase):
    def setUp(self):
        # Create a basic mock rule
        self.mock_rule = MagicMock(spec=Rule)
        self.mock_rule.rule_id = "test_rule"
        self.mock_rule.condition_logic = "AND"
        self.mock_rule.conditions = []

        self.sample_record = {
            "sku_name": "Premium Vanilla Ice Cream 1L",
            "bt": "Ice Cream Tub",
            "gk": ["Dairy", "Ice Cream", "Vanilla"],
            "category": "Frozen Desserts",
            "region": "Colombo",
            "price": 1500.00
        }

    def test_evaluate_single_condition(self):
        # sku_contains
        cond = {"condition_type": "sku_contains", "value": "vanilla"}
        self.assertTrue(_evaluate_single_condition(cond, self.sample_record))

        cond = {"condition_type": "sku_contains", "value": "chocolate"}
        self.assertFalse(_evaluate_single_condition(cond, self.sample_record))

        # bt_is
        cond = {"condition_type": "bt_is", "value": "Ice Cream Tub"}
        self.assertTrue(_evaluate_single_condition(cond, self.sample_record))

        # gk_contains
        cond = {"condition_type": "gk_contains", "value": "Dairy"}
        self.assertTrue(_evaluate_single_condition(cond, self.sample_record))

        # price_below / price_above
        cond = {"condition_type": "price_below", "value": "2000"}
        self.assertTrue(_evaluate_single_condition(cond, self.sample_record))
        cond = {"condition_type": "price_above", "value": "1000"}
        self.assertTrue(_evaluate_single_condition(cond, self.sample_record))

        # negate
        cond = {"condition_type": "sku_contains", "value": "chocolate", "negate": True}
        self.assertTrue(_evaluate_single_condition(cond, self.sample_record))

    def test_empty_conditions(self):
        self.mock_rule.conditions = []
        self.assertTrue(evaluate_conditions(self.mock_rule, self.sample_record))

    def test_or_logic_within_same_type_same_group(self):
        # If we have multiple 'sku_contains' in the same group, it should act as OR
        self.mock_rule.conditions = [
            {"condition_group": 1, "condition_type": "sku_contains", "value": "chocolate"}, # False
            {"condition_group": 1, "condition_type": "sku_contains", "value": "vanilla"}    # True
        ]
        self.assertTrue(evaluate_conditions(self.mock_rule, self.sample_record))

    def test_and_logic_different_types_same_group(self):
        # Different condition types in the same group act as AND
        self.mock_rule.conditions = [
            {"condition_group": 1, "condition_type": "sku_contains", "value": "vanilla"}, # True
            {"condition_group": 1, "condition_type": "price_below", "value": "1000"}     # False
        ]
        self.assertFalse(evaluate_conditions(self.mock_rule, self.sample_record))

        self.mock_rule.conditions[1]["value"] = "2000" # Make it True
        self.assertTrue(evaluate_conditions(self.mock_rule, self.sample_record))

    def test_cross_group_logic_and(self):
        self.mock_rule.condition_logic = "AND"
        self.mock_rule.conditions = [
            {"condition_group": 1, "condition_type": "sku_contains", "value": "vanilla"}, # Group 1 True
            {"condition_group": 2, "condition_type": "bt_is", "value": "Beverage"}        # Group 2 False
        ]
        self.assertFalse(evaluate_conditions(self.mock_rule, self.sample_record))

    def test_cross_group_logic_or(self):
        self.mock_rule.condition_logic = "OR"
        self.mock_rule.conditions = [
            {"condition_group": 1, "condition_type": "sku_contains", "value": "vanilla"}, # Group 1 True
            {"condition_group": 2, "condition_type": "bt_is", "value": "Beverage"}        # Group 2 False
        ]
        self.assertTrue(evaluate_conditions(self.mock_rule, self.sample_record))
    def test_flavor_conditions(self):
        # Setup mock flavor cache
        import engine.rules_engine.evaluator as eval_mod
        eval_mod._FLAVOR_CACHE = {
            "flavors_dict": {
                "chicken": "chicken",
                "chicken breast": "chicken",
                "spinach": "spinach",
                "shrimp": "prawn",
                "prawn": "prawn",
                "beef": "beef"
            },
            "meat_flavors": {"chicken", "beef"},
            "vegetable_flavors": {"spinach"},
            "seafood_flavors": {"prawn"}
        }

        # 1. flavor_contains
        # Match canonical name
        self.sample_record["sku_name"] = "Tasty Chicken Curry"
        cond = {"condition_type": "flavor_contains", "value": "chicken"}
        self.assertTrue(_evaluate_single_condition(cond, self.sample_record))

        # Match alias
        self.sample_record["sku_name"] = "Tasty Chicken Breast Soup"
        cond = {"condition_type": "flavor_contains", "value": "chicken"}
        self.assertTrue(_evaluate_single_condition(cond, self.sample_record))

        # Match target alias to canonical
        self.sample_record["sku_name"] = "Tasty Shrimp Salad"
        cond = {"condition_type": "flavor_contains", "value": "prawn"}
        self.assertTrue(_evaluate_single_condition(cond, self.sample_record))
        
        # Test mismatch
        self.sample_record["sku_name"] = "Tasty Beef Burger"
        cond = {"condition_type": "flavor_contains", "value": "spinach"}
        self.assertFalse(_evaluate_single_condition(cond, self.sample_record))

        # 2. flavor_is
        # Meat category
        self.sample_record["sku_name"] = "Tasty Beef Burger"
        cond = {"condition_type": "flavor_is", "value": "meat"}
        self.assertTrue(_evaluate_single_condition(cond, self.sample_record))

        # Vegetable category
        self.sample_record["sku_name"] = "Spinach Noodles 500g"
        cond = {"condition_type": "flavor_is", "value": "vegetable"}
        self.assertTrue(_evaluate_single_condition(cond, self.sample_record))

        # Vegetable category (using "veg" shorthand)
        cond = {"condition_type": "flavor_is", "value": "veg"}
        self.assertTrue(_evaluate_single_condition(cond, self.sample_record))

        # Seafood category
        self.sample_record["sku_name"] = "Fresh Shrimp"
        cond = {"condition_type": "flavor_is", "value": "seafood"}
        self.assertTrue(_evaluate_single_condition(cond, self.sample_record))

        # Mismatch category
        self.sample_record["sku_name"] = "Spinach Noodles 500g"
        cond = {"condition_type": "flavor_is", "value": "meat"}
        self.assertFalse(_evaluate_single_condition(cond, self.sample_record))

if __name__ == '__main__':
    unittest.main()
