"""SQLAlchemy ORM models for catalog items, brands/flavors, and taxonomy dictionaries."""

from sqlalchemy import Column, Float, Index, Integer, Text

from backend.models.base_model import Base


class CatalogItem(Base):
    """ORM definition for the `catalog_items` reference table."""

    __tablename__ = "catalog_items"

    id = Column("id", Integer, primary_key=True)
    domain = Column("domain", Text, nullable=False)
    name = Column("name", Text, nullable=False)
    brand = Column("brand", Text)
    flavor = Column("flavor", Text)
    basictype = Column("basictype", Text)
    generic_keywords = Column("generic_keywords", Text)
    price = Column("price", Float)
    category = Column("category", Text)
    region = Column("region", Text)
    description = Column("description", Text)
    clean_text = Column("clean_text", Text)
    # JSON text: the (value, unit, type) weight tuple, or a bare number expanded on read
    weight_val = Column("weight_val", Text)
    entities_json = Column("entities_json", Text)
    token_count = Column("token_count", Integer)
    clean_no_weights = Column("clean_no_weights", Text)
    row_hash = Column("row_hash", Text, nullable=False)

    __table_args__ = (
        Index("idx_catalog_items_domain_hash", "domain", "row_hash"),
        Index(
            "catalog_search_trgm_idx",
            "name",
            "description",
            postgresql_using="gin",
            postgresql_ops={"name": "gin_trgm_ops", "description": "gin_trgm_ops"},
        ),
    )


class BrandFlavor(Base):
    """ORM definition for the `brand_flavors` reference table."""

    __tablename__ = "brand_flavors"

    id = Column("id", Integer, primary_key=True)
    domain = Column("domain", Text, nullable=False)
    name = Column("name", Text, nullable=False)
    aliases = Column("aliases", Text)
    is_weak = Column("is_weak", Integer, server_default="0")
    is_meat = Column("is_meat", Integer, server_default="0")
    is_vegetable = Column("is_vegetable", Integer, server_default="0")
    is_seafood = Column("is_seafood", Integer, server_default="0")
    catalog_count = Column("catalog_count", Integer, server_default="0")
    row_hash = Column("row_hash", Text, nullable=False)

    __table_args__ = (Index("idx_brand_flavors_count", "domain", "catalog_count"),)


class ClassifierDictionary(Base):
    """ORM definition for the `classifier_dictionaries` reference table."""

    __tablename__ = "classifier_dictionaries"

    id = Column("id", Integer, primary_key=True)
    domain = Column("domain", Text, nullable=False)
    tag_type = Column("tag_type", Text, nullable=False)
    tag = Column("tag", Text, nullable=False)
    catalog_count = Column("catalog_count", Integer, server_default="0")
    metadata_json = Column("metadata_json", Text)

    __table_args__ = (
        Index("idx_classifier_dict_lookup", "domain", "tag_type", "catalog_count"),
    )


class BtGkMap(Base):
    """ORM definition for the `bt_gk_map` reference table."""

    __tablename__ = "bt_gk_map"

    id = Column("id", Integer, primary_key=True)
    domain = Column("domain", Text, nullable=False)
    basictype = Column("basictype", Text, nullable=False)
    generic_keywords = Column("generic_keywords", Text, nullable=False)
    gk_count = Column("gk_count", Integer, server_default="0")
    catalog_count = Column("catalog_count", Integer, server_default="0")

    __table_args__ = (
        Index("idx_bt_gk_map_domain", "domain", "basictype"),
        Index("idx_bt_gk_map_count", "domain", "catalog_count"),
    )


catalog_items = CatalogItem.__table__
brand_flavors = BrandFlavor.__table__
classifier_dictionaries = ClassifierDictionary.__table__
bt_gk_map = BtGkMap.__table__
