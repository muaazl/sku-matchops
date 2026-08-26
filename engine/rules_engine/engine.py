import logging
from .loader import get_rules
from .evaluator import evaluate_conditions
from .actions import apply_actions

logger = logging.getLogger("matchops.rules_engine")

def run_rules_engine(record: dict) -> dict:
    """
    Main orchestrator for the Post-Processing Rules Engine.
    Receives a structured record, applies rules, and returns augmented record.
    """
    domain = record.get("domain")
    if not domain:
        logger.warning("No domain found in record, skipping rules engine.")
        return record
        
    rules_applied = list(record.get("rules_applied") or [])
    
    # Run modules in fixed sequence
    modules = ["bt_override", "gk_injection", "formatter", "visibility"]
    
    for module in modules:
        rules = get_rules(domain=domain, module=module)
        
        for rule in rules:
            if evaluate_conditions(rule, record):
                changes = apply_actions(rule, record)
                if changes: # Only log if it actually changed something
                    rules_applied.append({
                        "rule_id": rule.rule_id,
                        "module": module,
                        "description": rule.description,
                        "change": changes,
                        "reasoning": rule.reasoning
                    })
                    
    record["rules_applied"] = rules_applied
    return record
