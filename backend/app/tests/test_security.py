import unittest
from pydantic import ValidationError
from backend.app.api.endpoints.catalog import escape_meili_filter_value
from backend.app.schemas.models import RuleModel, RuleConditionModel, RuleActionModel

class TestSecurityHardening(unittest.TestCase):
    
    def test_meili_filter_escaping(self):
        # Escape simple values
        self.assertEqual(escape_meili_filter_value("Colombo"), "Colombo")
        
        # Escape double quotes
        self.assertEqual(escape_meili_filter_value('food" OR price >= 0'), 'food\\" OR price >= 0')
        
        # Escape backslashes
        self.assertEqual(escape_meili_filter_value("a\\b"), "a\\\\b")
        
        # Escape mixed characters
        self.assertEqual(escape_meili_filter_value('a\\"b'), 'a\\\\\\"b')

    def test_pydantic_rule_validation_valid(self):
        # Valid rule schema should not raise validation errors
        valid_payload = {
            "rule_id": "rule_123",
            "domain": "market",
            "module": "bt_override",
            "priority": 100,
            "description": "A valid rule description.",
            "reasoning": "Reasoning behind this rule.",
            "condition_logic": "AND",
            "is_active": 1,
            "conditions": [
                {
                    "condition_group": 1,
                    "condition_type": "sku_contains",
                    "value": "coke",
                    "negate": 0
                }
            ],
            "actions": [
                {
                    "action_type": "set_bt",
                    "value": "Soft Drink"
                }
            ]
        }
        try:
            RuleModel(**valid_payload)
        except ValidationError as e:
            self.fail(f"ValidationError raised unexpectedly for valid payload: {e}")

    def test_pydantic_rule_validation_invalid_domain(self):
        # Invalid domain name should be rejected
        invalid_payload = {
            "rule_id": "rule_123",
            "domain": "invalid_domain",  # Rejected
            "module": "bt_override",
            "priority": 100,
            "description": "Description",
            "reasoning": "Reasoning",
            "conditions": [],
            "actions": []
        }
        with self.assertRaises(ValidationError):
            RuleModel(**invalid_payload)

    def test_pydantic_rule_validation_invalid_module(self):
        # Invalid module name should be rejected
        invalid_payload = {
            "rule_id": "rule_123",
            "domain": "market",
            "module": "invalid_module",  # Rejected
            "priority": 100,
            "description": "Description",
            "reasoning": "Reasoning",
            "conditions": [],
            "actions": []
        }
        with self.assertRaises(ValidationError):
            RuleModel(**invalid_payload)

    def test_pydantic_rule_validation_excessive_lengths(self):
        # Description/Reasoning fields exceeding length limits should be rejected
        long_description = "A" * 251  # Max limit is 250
        invalid_payload = {
            "rule_id": "rule_123",
            "domain": "market",
            "module": "bt_override",
            "priority": 100,
            "description": long_description,
            "reasoning": "Reasoning",
            "conditions": [],
            "actions": []
        }
        with self.assertRaises(ValidationError):
            RuleModel(**invalid_payload)

    def test_pydantic_rule_validation_invalid_condition_type(self):
        # Invalid condition type should be rejected
        invalid_payload = {
            "rule_id": "rule_123",
            "domain": "market",
            "module": "bt_override",
            "priority": 100,
            "description": "Description",
            "reasoning": "Reasoning",
            "conditions": [
                {
                    "condition_group": 1,
                    "condition_type": "invalid_type",  # Rejected
                    "value": "test"
                }
            ],
            "actions": []
        }
        with self.assertRaises(ValidationError):
            RuleModel(**invalid_payload)

    def test_pydantic_rule_validation_invalid_action_type(self):
        # Invalid action type should be rejected
        invalid_payload = {
            "rule_id": "rule_123",
            "domain": "market",
            "module": "bt_override",
            "priority": 100,
            "description": "Description",
            "reasoning": "Reasoning",
            "conditions": [],
            "actions": [
                {
                    "action_type": "invalid_action",  # Rejected
                    "value": "test"
                }
            ]
        }
        with self.assertRaises(ValidationError):
            RuleModel(**invalid_payload)
