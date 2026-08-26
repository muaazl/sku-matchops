import unittest
from unittest.mock import MagicMock, patch

from engine.rules_engine.actions import apply_actions
from engine.rules_engine.engine import run_rules_engine
from engine.rules_engine.loader import Rule


class TestRulesActions(unittest.TestCase):
    def _create_rule(self, actions: list, rule_id: str = "r1", module: str = "bt_override", description: str = "desc", reasoning: str = "reason") -> Rule:
        rule = MagicMock(spec=Rule)
        rule.rule_id = rule_id
        rule.module = module
        rule.description = description
        rule.reasoning = reasoning
        rule.actions = actions
        return rule

    def test_set_bt(self):
        rule = self._create_rule([{"action_type": "set_bt", "value": "Ice Cream Tub"}])
        record = {"bt": "Ice Cream Cone"}
        changes = apply_actions(rule, record)
        self.assertEqual(record["bt"], "Ice Cream Tub")
        self.assertIn("BT changed to 'Ice Cream Tub'", changes)

        # Same value should be a no-op and not log a change
        changes_noop = apply_actions(rule, record)
        self.assertEqual(changes_noop, "")

    def test_add_gk(self):
        rule = self._create_rule([{"action_type": "add_gk", "value": "Vanilla"}])

        # Normal list
        record = {"gk": ["Dairy"]}
        changes = apply_actions(rule, record)
        self.assertEqual(record["gk"], ["Dairy", "Vanilla"])
        self.assertIn("Added GK 'Vanilla'", changes)

        # Duplicate should be a no-op
        changes_dup = apply_actions(rule, record)
        self.assertEqual(record["gk"], ["Dairy", "Vanilla"])
        self.assertEqual(changes_dup, "")

        # Missing 'gk' key
        record_missing = {}
        changes_missing = apply_actions(rule, record_missing)
        self.assertEqual(record_missing["gk"], ["Vanilla"])
        self.assertIn("Added GK 'Vanilla'", changes_missing)

        # 'gk' is None
        record_none = {"gk": None}
        changes_none = apply_actions(rule, record_none)
        self.assertEqual(record_none["gk"], ["Vanilla"])
        self.assertIn("Added GK 'Vanilla'", changes_none)

    def test_remove_gk(self):
        rule = self._create_rule([{"action_type": "remove_gk", "value": "Vanilla"}])

        # Normal list containing item
        record = {"gk": ["Dairy", "Vanilla", "Tub"]}
        changes = apply_actions(rule, record)
        self.assertEqual(record["gk"], ["Dairy", "Tub"])
        self.assertIn("Removed GK 'Vanilla'", changes)

        # Item not in list is a no-op
        changes_noop = apply_actions(rule, record)
        self.assertEqual(record["gk"], ["Dairy", "Tub"])
        self.assertEqual(changes_noop, "")

        # 'gk' is None
        record_none = {"gk": None}
        changes_none = apply_actions(rule, record_none)
        self.assertEqual(changes_none, "")

    def test_set_region_and_category(self):
        rule_reg = self._create_rule([{"action_type": "set_region", "value": "Colombo"}])
        record = {"region": "Kandy"}
        changes_reg = apply_actions(rule_reg, record)
        self.assertEqual(record["region"], "Colombo")
        self.assertIn("Region changed to 'Colombo'", changes_reg)

        rule_cat = self._create_rule([{"action_type": "set_category", "value": "Frozen"}])
        changes_cat = apply_actions(rule_cat, record)
        self.assertEqual(record["category"], "Frozen")
        self.assertIn("Category changed to 'Frozen'", changes_cat)

    def test_set_visibility(self):
        rule = self._create_rule([{"action_type": "set_visibility", "value": "hidden"}])
        record = {"visibility": "visible"}
        changes = apply_actions(rule, record)
        self.assertEqual(record["visibility"], "hidden")
        self.assertIn("Visibility set to 'hidden'", changes)

    def test_normalize_sku(self):
        rule = self._create_rule([{"action_type": "normalize_sku", "value": "Icecream|Ice Cream"}])

        record = {"sku_name": "Vanilla Icecream 1L"}
        changes = apply_actions(rule, record)
        self.assertEqual(record["sku_name"], "Vanilla Ice Cream 1L")
        self.assertIn("Normalized SKU substring 'Icecream' to 'Ice Cream'", changes)

        # Substring not found
        record_noop = {"sku_name": "Vanilla Ice Cream 1L"}
        changes_noop = apply_actions(rule, record_noop)
        self.assertEqual(changes_noop, "")

        # None sku_name
        record_none = {"sku_name": None}
        changes_none = apply_actions(rule, record_none)
        self.assertEqual(changes_none, "")

        # Malformed action value without pipe
        rule_bad = self._create_rule([{"action_type": "normalize_sku", "value": "Icecream"}])
        changes_bad = apply_actions(rule_bad, record)
        self.assertEqual(changes_bad, "")

    def test_compound_actions(self):
        rule = self._create_rule([
            {"action_type": "set_bt", "value": "Ice Cream Tub"},
            {"action_type": "add_gk", "value": "Dessert"},
            {"action_type": "set_visibility", "value": "visible"}
        ])
        record = {"bt": "Cone", "gk": ["Dairy"], "visibility": "draft"}
        changes = apply_actions(rule, record)

        self.assertEqual(record["bt"], "Ice Cream Tub")
        self.assertEqual(record["gk"], ["Dairy", "Dessert"])
        self.assertEqual(record["visibility"], "visible")
        self.assertEqual(changes, "BT changed to 'Ice Cream Tub'; Added GK 'Dessert'; Visibility set to 'visible'")


class TestRulesEngineOrchestrator(unittest.TestCase):
    def test_missing_domain_early_exit(self):
        record = {"sku_name": "Test SKU"}
        result = run_rules_engine(record)
        self.assertEqual(result, record)
        self.assertNotIn("rules_applied", result)

    @patch("engine.rules_engine.engine.get_rules")
    @patch("engine.rules_engine.engine.evaluate_conditions")
    def test_fixed_module_execution_sequence(self, mock_eval, mock_get_rules):
        # Setup mock rules for each module
        modules_called = []

        def side_effect_get_rules(domain, module):
            modules_called.append(module)
            r = MagicMock(spec=Rule)
            r.rule_id = f"rule_{module}"
            r.module = module
            r.description = f"Desc for {module}"
            r.reasoning = f"Reason for {module}"
            r.actions = [{"action_type": "set_visibility", "value": module}]
            return [r]

        mock_get_rules.side_effect = side_effect_get_rules
        mock_eval.return_value = True

        record = {"domain": "food", "sku_name": "Test SKU"}
        result = run_rules_engine(record)

        # Expected fixed sequence: BT Override -> GK Injection -> Formatter -> Visibility
        self.assertEqual(modules_called, ["bt_override", "gk_injection", "formatter", "visibility"])
        self.assertEqual(len(result["rules_applied"]), 4)
        self.assertEqual([r["module"] for r in result["rules_applied"]], ["bt_override", "gk_injection", "formatter", "visibility"])

    @patch("engine.rules_engine.engine.get_rules")
    @patch("engine.rules_engine.engine.evaluate_conditions")
    def test_rules_applied_provenance_and_preservation(self, mock_eval, mock_get_rules):
        r1 = MagicMock(spec=Rule)
        r1.rule_id = "R_BT_01"
        r1.module = "bt_override"
        r1.description = "Override Cone to Tub"
        r1.reasoning = "Catalog standard"
        r1.actions = [{"action_type": "set_bt", "value": "Tub"}]

        def side_effect(domain, module):
            return [r1] if module == "bt_override" else []

        mock_get_rules.side_effect = side_effect
        mock_eval.return_value = True

        existing_audit = [{"rule_id": "PREV_01", "module": "pre", "description": "prior", "change": "none", "reasoning": "test"}]
        record = {"domain": "market", "bt": "Cone", "rules_applied": existing_audit}

        result = run_rules_engine(record)
        self.assertEqual(result["bt"], "Tub")
        self.assertEqual(len(result["rules_applied"]), 2)
        self.assertEqual(result["rules_applied"][0]["rule_id"], "PREV_01")
        self.assertEqual(result["rules_applied"][1]["rule_id"], "R_BT_01")
        self.assertEqual(result["rules_applied"][1]["change"], "BT changed to 'Tub'")


if __name__ == "__main__":
    unittest.main()
