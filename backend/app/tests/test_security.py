import unittest

from pydantic import ValidationError

from backend.app.schemas.models import RuleModel


class TestSecurityHardening(unittest.TestCase):
    def test_pydantic_rule_validation_valid(self):
        # Valid rule schema should not raise validation errors
        valid_payload = {
            "rule_id": "rule_123",
            "domain": "market",
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
                    "negate": 0,
                }
            ],
            "actions": [{"action_type": "set_bt", "value": "Soft Drink"}],
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
            "priority": 100,
            "description": "Description",
            "reasoning": "Reasoning",
            "conditions": [],
            "actions": [],
        }
        with self.assertRaises(ValidationError):
            RuleModel(**invalid_payload)

    def test_pydantic_rule_validation_excessive_lengths(self):
        # Description/Reasoning fields exceeding length limits should be rejected
        long_description = "A" * 251  # Max limit is 250
        invalid_payload = {
            "rule_id": "rule_123",
            "domain": "market",
            "priority": 100,
            "description": long_description,
            "reasoning": "Reasoning",
            "conditions": [],
            "actions": [],
        }
        with self.assertRaises(ValidationError):
            RuleModel(**invalid_payload)

    def test_pydantic_rule_validation_invalid_condition_type(self):
        # Invalid condition type should be rejected
        invalid_payload = {
            "rule_id": "rule_123",
            "domain": "market",
            "priority": 100,
            "description": "Description",
            "reasoning": "Reasoning",
            "conditions": [
                {
                    "condition_group": 1,
                    "condition_type": "invalid_type",  # Rejected
                    "value": "test",
                }
            ],
            "actions": [],
        }
        with self.assertRaises(ValidationError):
            RuleModel(**invalid_payload)

    def test_pydantic_rule_validation_invalid_action_type(self):
        # Invalid action type should be rejected
        invalid_payload = {
            "rule_id": "rule_123",
            "domain": "market",
            "priority": 100,
            "description": "Description",
            "reasoning": "Reasoning",
            "conditions": [],
            "actions": [
                {
                    "action_type": "invalid_action",  # Rejected
                    "value": "test",
                }
            ],
        }
        with self.assertRaises(ValidationError):
            RuleModel(**invalid_payload)
