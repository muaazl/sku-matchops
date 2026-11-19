from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from engine import config
from engine.classification.classifier import TaxonomyClassifier
from engine.classification.cold_start_router import TaxonomyLifecycleRegistry
from engine.data_pipeline.cache_manager import CacheManager, calculate_row_hash
from engine.nlp.text_cleaner import TextPipeline
from scripts import sync as sync_mod


def test_resolve_input_category_col_never_returns_target_columns():
    assert config.resolve_input_category_col(["Name", "SellerCategory", "category"]) == "SellerCategory"
    assert config.resolve_input_category_col(["Name", "seller_category", "region"]) == "seller_category"
    assert config.resolve_input_category_col(["Name", "Category", "category", "region"]) == ""


def test_classifier_train_and_centroids_use_seller_category_not_target(tmp_path):
    n = 16
    df = pd.DataFrame(
        {
            "Name": [f"Item {i}" for i in range(n)],
            "Description": [""] * n,
            "SellerCategory": [f"SELLER_CAT_{i}" for i in range(n)],
            "category": [f"TARGET_CAT_{i % 2}" for i in range(n)],
            "basictype": [f"BT_{i % 2}" for i in range(n)],
            "Generic keywords": [f"kw_{i % 2}" for i in range(n)],
            "Price": [100.0 + i for i in range(n)],
        }
    )

    mock_model = MagicMock()
    mock_model.encode.return_value = {"dense": np.ones((2, 4), dtype=np.float32)}

    captured_cats: list[list[str]] = []

    def fake_embed_weighted(names, descs, cats, weights=None):
        captured_cats.append(list(cats))
        return {"dense": np.ones((len(names), 4), dtype=np.float32)}

    mock_model.embed_weighted_sku.side_effect = fake_embed_weighted

    clf = TaxonomyClassifier.__new__(TaxonomyClassifier)
    clf.domain = config.DOMAIN_MARKET
    clf.cache_dir = tmp_path
    clf.model = mock_model
    clf.cat_df = df
    clf.bt_descs = {"BT_0": "b0", "BT_1": "b1"}
    clf.third_tag_descs = {"TARGET_CAT_0": "t0", "TARGET_CAT_1": "t1"}
    clf.third_tag_overrides = {}
    clf.bt_to_gk_umbrella = {}
    clf._trained = False

    orig_cache_dir = config.CACHE_DIR
    config.CACHE_DIR = tmp_path
    try:
        clf._try_train(force_retrain=True)
    finally:
        config.CACHE_DIR = orig_cache_dir

    assert len(captured_cats) == 1
    assert captured_cats[0] == [f"SELLER_CAT_{i}" for i in range(n)]

    # Verify ColdStartRouter centroid lookup uses the exact same SellerCategory key in weighted cache
    reg = TaxonomyLifecycleRegistry(domain="market", cache_dir=tmp_path, few_shot_threshold=2)
    reg.warm_classes = ["BT_0", "BT_1"]
    reg._build_or_load_centroids(df, embed_engine=None, force_rebuild=True)
    assert len(reg.warm_index_to_class) == 2


def test_cache_manager_and_arcface_loader_and_rerank_use_seller_category(tmp_path):
    df = pd.DataFrame(
        {
            "Name": [f"Dish {i}" for i in range(12)],
            "Description": [""] * 12,
            "SellerCategory": [f"MENU_CAT_{i}" for i in range(12)],
            "region": ["Sri Lankan"] * 12,
            "basictype": ["Curry"] * 12,
            "Generic keywords": ["Curry"] * 12,
            "Price": [500.0] * 12,
        }
    )

    # 1. CacheManager embeds SellerCategory (not region) into Qdrant
    mock_embed = MagicMock()
    mock_embed.has_learned_sparse = False
    mock_embed.embed_weighted_sku.return_value = {
        "dense": np.ones((len(df), 4), dtype=np.float32),
        "bm25": [{} for _ in range(len(df))],
        "sparse": None,
    }
    mock_ner = MagicMock()
    mock_ner.target_label = "flavor"
    mock_ner.batch_extract_entities.side_effect = lambda texts, batch_size=32: [{} for _ in texts]

    orig_cache_dir = config.CACHE_DIR
    config.CACHE_DIR = tmp_path
    try:
        with patch("engine.data_pipeline.vector_store.VectorStore") as mock_vs_cls:
            mock_vs = mock_vs_cls.return_value
            mock_vs._get_collection_name.return_value = "food_catalog"
            mock_vs.client.collection_exists.return_value = False
            cm = CacheManager(mock_ner, mock_embed)
            cm.manage_catalog_cache(df, pd.DataFrame(), domain="food", force_sync=True)
    finally:
        config.CACHE_DIR = orig_cache_dir

    cats_sent = mock_embed.embed_weighted_sku.call_args[0][2]
    assert cats_sent == [f"menu cat {i}" for i in range(12)]
    assert "sri lankan" not in cats_sent

    # 2. ArcFace load_catalog_data picks SellerCategory (not region/category)
    with patch("engine.data_pipeline.ingestion.DataIngestion.load_catalog", return_value=(df, pd.DataFrame())):
        clean_df = sync_mod.load_catalog_data("food", from_sample=False)
    assert clean_df["Category"].tolist() == [f"MENU_CAT_{i}" for i in range(12)]

    # 3. Reranker formatter uses SellerCategory and ignores target region/category
    assert "SellerCategory:" not in TextPipeline.format_catalog_row_for_reranker(
        {"Name": "Fish Curry", "region": "Sri Lankan", "category": "Groceries"},
        domain="food",
    )
    assert "SellerCategory: CURRIES" in TextPipeline.format_catalog_row_for_reranker(
        {"Name": "Fish Curry", "SellerCategory": "CURRIES", "region": "Sri Lankan"},
        domain="food",
    )

    # 4. Row hash detects SellerCategory changes
    h1 = calculate_row_hash({"Name": "A", "SellerCategory": "Cat1"}, domain="food")
    h2 = calculate_row_hash({"Name": "A", "SellerCategory": "Cat2"}, domain="food")
    assert h1 != h2
