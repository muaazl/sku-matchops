"""SQLAlchemy ORM models for deterministic business rules, conditions, and actions."""

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, Text, func

from backend.models.base_model import Base


class Rule(Base):
    """ORM definition for the `rules` table."""

    __tablename__ = "rules"

    id = Column("id", Integer, primary_key=True)
    rule_id = Column("rule_id", Text, nullable=False, unique=True)
    domain = Column("domain", Text, nullable=False)
    priority = Column("priority", Integer, nullable=False, server_default="100")
    description = Column("description", Text, nullable=False)
    reasoning = Column("reasoning", Text, nullable=False)
    condition_logic = Column("condition_logic", Text, nullable=False, server_default="AND")
    is_active = Column("is_active", Integer, nullable=False, server_default="1")
    created_at = Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (Index("idx_rules_domain_priority", "domain", "priority"),)


class RuleCondition(Base):
    """ORM definition for the `conditions` table."""

    __tablename__ = "conditions"

    id = Column("id", Integer, primary_key=True)
    rule_id = Column(
        "rule_id",
        Text,
        ForeignKey("rules.rule_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    condition_group = Column("condition_group", Integer, nullable=False, server_default="1")
    condition_type = Column("condition_type", Text, nullable=False)
    value = Column("value", Text, nullable=False)
    negate = Column("negate", Integer, nullable=False, server_default="0")
    operator = Column("operator", Text, nullable=False, server_default="AND")


class RuleAction(Base):
    """ORM definition for the `actions` table."""

    __tablename__ = "actions"

    id = Column("id", Integer, primary_key=True)
    rule_id = Column(
        "rule_id",
        Text,
        ForeignKey("rules.rule_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    action_type = Column("action_type", Text, nullable=False)
    value = Column("value", Text, nullable=False)


rules = Rule.__table__
conditions = RuleCondition.__table__
actions = RuleAction.__table__
