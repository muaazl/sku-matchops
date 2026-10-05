import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from engine import config
from engine.classification.classifier import ZeroShotClassifier
from engine.classification.models.arcface_bt import BTArcFaceNet, FocalLoss


class TestArcFaceBT(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.orig_onnx_dir = config.ONNX_DIR
        self.orig_cache_dir = config.CACHE_DIR
        self.orig_arcface_dir = getattr(config, "ARCFACE_DIR", None)
        self.orig_food_model = config.FOOD_BT_MODEL
        self.orig_market_model = config.MARKET_BT_MODEL

        config.ONNX_DIR = self.test_dir
        config.CACHE_DIR = self.test_dir
        config.ARCFACE_DIR = self.test_dir

    def tearDown(self):
        config.ONNX_DIR = self.orig_onnx_dir
        config.CACHE_DIR = self.orig_cache_dir
        if self.orig_arcface_dir is not None:
            config.ARCFACE_DIR = self.orig_arcface_dir
        config.FOOD_BT_MODEL = self.orig_food_model
        config.MARKET_BT_MODEL = self.orig_market_model
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_bt_arcface_net_forward_shapes(self):
        """Test BTArcFaceNet forward pass for training and inference."""
        batch_size = 8
        num_classes = 10
        model = BTArcFaceNet(
            in_features=1025, hidden_features=512, num_classes=num_classes, scale=30.0, margin=0.35
        )

        # Training forward pass with labels
        model.train()
        x_train = torch.randn(batch_size, 1025)
        labels = torch.randint(0, num_classes, (batch_size,))
        logits_train = model(x_train, labels=labels)
        self.assertEqual(logits_train.shape, (batch_size, num_classes))

        # Evaluation forward pass without labels
        model.eval()
        x_eval = torch.randn(batch_size, 1025)
        logits_eval = model(x_eval)
        self.assertEqual(logits_eval.shape, (batch_size, num_classes))

        # Latent feature extraction
        latent = model.extract_features(x_eval)
        self.assertEqual(latent.shape, (batch_size, 512))

    def test_focal_loss_behavior(self):
        """Test FocalLoss computation and long-tail error penalization."""
        criterion = FocalLoss(gamma=2.0, reduction="mean")

        # Well-separated confident logits -> low loss
        easy_logits = torch.tensor([[10.0, -5.0, -5.0, -5.0, -5.0]])
        target_easy = torch.tensor([0])
        easy_loss = criterion(easy_logits, target_easy).item()

        # Inverted ambiguous logits -> high loss
        hard_logits = torch.tensor([[-5.0, 10.0, -5.0, -5.0, -5.0]])
        target_hard = torch.tensor([0])
        hard_loss = criterion(hard_logits, target_hard).item()

        self.assertGreater(hard_loss, easy_loss)
        self.assertLess(easy_loss, 0.1)

    def test_onnx_export_and_int8_quantization(self):
        """Test export to ONNX evaluation graph, INT8 quantization, and ORT execution with mmap."""
        num_classes = 4
        model = BTArcFaceNet(in_features=1025, hidden_features=512, num_classes=num_classes)
        model.eval()

        fp32_path = os.path.join(self.test_dir, "test_bt_arcface.onnx")
        int8_path = os.path.join(self.test_dir, "test_bt_arcface_int8.onnx")

        # 1. Export to ONNX
        dummy_input = torch.randn(1, 1025, dtype=torch.float32)
        torch.onnx.export(
            model,
            dummy_input,
            fp32_path,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
        self.assertTrue(os.path.exists(fp32_path))

        # 2. Dynamic INT8 Quantization
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(
            model_input=fp32_path,
            model_output=int8_path,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
        )
        self.assertTrue(os.path.exists(int8_path))

        # 3. Load via ORT with mmap configuration
        sess_opts = ort.SessionOptions()
        sess_opts.add_session_config_entry("session.use_mmap_for_weights", "1")
        session = ort.InferenceSession(
            int8_path, sess_options=sess_opts, providers=["CPUExecutionProvider"]
        )

        test_inputs = np.random.randn(3, 1025).astype(np.float32)
        ort_out = session.run(None, {session.get_inputs()[0].name: test_inputs})[0]
        self.assertEqual(ort_out.shape, (3, num_classes))

    def test_fail_fast_missing_model_exception(self):
        """Service must fail fast with descriptive exception if ArcFace model/labels are missing."""
        config.FOOD_BT_MODEL = "arcface"

        clf = ZeroShotClassifier.__new__(ZeroShotClassifier)
        clf.domain = config.DOMAIN_FOOD
        clf.bt_model = "arcface"

        with self.assertRaises(RuntimeError) as ctx:
            clf._load_arcface_model()

        self.assertIn("[FAIL-FAST]", str(ctx.exception))
        self.assertIn("missing", str(ctx.exception).lower())

    def test_fail_fast_corrupt_labels_exception(self):
        """Service must fail fast if label mapping is corrupt."""
        config.FOOD_BT_MODEL = "arcface"

        # Create dummy onnx file
        onnx_file = os.path.join(self.test_dir, "food_bt_arcface_int8.onnx")
        with open(onnx_file, "wb") as f:
            f.write(b"corrupt_onnx_bytes")

        # Create corrupt labels file
        labels_file = os.path.join(self.test_dir, "food_bt_arcface_labels.json")
        with open(labels_file, "w") as f:
            f.write("{invalid_json:")

        clf = ZeroShotClassifier.__new__(ZeroShotClassifier)
        clf.domain = config.DOMAIN_FOOD
        clf.bt_model = "arcface"

        with self.assertRaises(RuntimeError) as ctx:
            clf._load_arcface_model()

        self.assertIn("[FAIL-FAST]", str(ctx.exception))

    def test_arcface_prediction_and_zero_shot_fallback(self):
        """Test ArcFace prediction routing, thresholding, and zero-shot fallback."""
        clf = ZeroShotClassifier.__new__(ZeroShotClassifier)
        clf.domain = config.DOMAIN_FOOD
        clf.bt_model = "arcface"
        clf.cache_dir = self.test_dir
        clf._trained = True

        clf._arcface_classes = ["Pizza", "Burger", "Beverage"]
        clf._arcface_input_name = "input"

        # Setup fitted scaler
        scaler = StandardScaler()
        scaler.mean_ = np.array([2.0], dtype=np.float32)
        scaler.scale_ = np.array([0.5], dtype=np.float32)
        scaler.var_ = np.array([0.25], dtype=np.float32)
        clf._arcface_price_scaler = scaler

        # Zero-shot fallback descriptions
        clf.bt_labels = ["Pizza", "Burger", "Beverage"]
        clf.bt_embs_pure = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
        clf.bt_embs_desc = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)

        # 1. High confidence prediction
        mock_sess = MagicMock()
        # High logit for Pizza (index 0) -> confidence ~0.90
        mock_sess.run.return_value = [np.array([[5.0, 0.0, -1.0]], dtype=np.float32)]
        clf._arcface_session = mock_sess

        test_vec = np.array([1.0, 0.0], dtype=np.float32)
        pred, conf, source, _ = clf.predict_bt(test_vec, price=150.0)

        self.assertEqual(pred, "Pizza")
        self.assertGreaterEqual(conf, 0.40)
        self.assertEqual(source, "trained")
        self.assertEqual(clf.active_bt_model, "arcface")

        # 2. Low confidence (< 0.40) -> falls back to zero-shot
        # Uniform logits -> confidence 1/3 = 0.333 (< 0.40)
        mock_sess.run.return_value = [np.array([[1.0, 1.0, 1.0]], dtype=np.float32)]
        pred_zs, conf_zs, source_zs, _ = clf.predict_bt(test_vec, price=150.0)

        self.assertEqual(source_zs, "zero-shot")
        self.assertEqual(pred_zs, "Pizza")  # Matched via cosine similarity in zero-shot

    def test_logreg_model_toggle_behavior(self):
        """When domain is configured for 'logreg', LogReg model must execute without loading ArcFace."""
        clf = ZeroShotClassifier.__new__(ZeroShotClassifier)
        clf.domain = config.DOMAIN_MARKET
        clf.bt_model = "logreg"
        clf.cache_dir = self.test_dir
        clf._trained = True

        # Mock LogReg estimator
        mock_clf = MagicMock()
        mock_clf.predict_proba.return_value = np.array([[0.1, 0.85, 0.05]])
        clf._bt_clf = mock_clf

        mock_enc = MagicMock()
        mock_enc.classes_ = np.array(["Shampoo", "Soap", "Toothpaste"])
        clf._bt_enc = mock_enc

        clf._price_scaler = None
        clf._arcface_session = None

        test_vec = np.zeros(1024, dtype=np.float32)
        pred, conf, source, _ = clf.predict_bt(test_vec, price=50.0)

        self.assertEqual(pred, "Soap")
        self.assertAlmostEqual(conf, 0.85)
        self.assertEqual(source, "trained")
        self.assertEqual(clf.active_bt_model, "logreg")

    def test_batch_predict_bt_arcface(self):
        """Test batch_predict_bt routing with ArcFace."""
        clf = ZeroShotClassifier.__new__(ZeroShotClassifier)
        clf.domain = config.DOMAIN_FOOD
        clf.bt_model = "arcface"
        clf.cache_dir = self.test_dir
        clf._trained = True

        clf._arcface_classes = ["Pizza", "Burger", "Beverage"]
        clf._arcface_input_name = "input"

        scaler = StandardScaler()
        scaler.mean_ = np.array([2.0], dtype=np.float32)
        scaler.scale_ = np.array([0.5], dtype=np.float32)
        scaler.var_ = np.array([0.25], dtype=np.float32)
        clf._arcface_price_scaler = scaler

        clf.bt_labels = ["Pizza", "Burger", "Beverage"]
        clf.bt_embs_pure = np.eye(3, dtype=np.float32)
        clf.bt_embs_desc = np.eye(3, dtype=np.float32)

        mock_sess = MagicMock()
        # Item 0: confident Pizza, Item 1: ambiguous (falls back to zero-shot Burger)
        mock_sess.run.return_value = [
            np.array(
                [
                    [5.0, 0.0, -1.0],  # item 0
                    [1.0, 1.0, 1.0],  # item 1
                ],
                dtype=np.float32,
            )
        ]
        clf._arcface_session = mock_sess

        test_vecs = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        results = clf.batch_predict_bt(test_vecs, [100.0, 200.0])
        self.assertEqual(len(results), 2)
        # Item 0: trained arcface
        self.assertEqual(results[0][0], "Pizza")
        self.assertEqual(results[0][2], "trained")
        # Item 1: zero-shot fallback
        self.assertEqual(results[1][0], "Burger")
        self.assertEqual(results[1][2], "zero-shot")

    def test_tagger_metadata_includes_active_model(self):
        """Tagger batch classification output dicts must include 'model' and 'bt_model'."""
        from engine.classification.tagger import tag_all_skus

        clf = ZeroShotClassifier.__new__(ZeroShotClassifier)
        clf.domain = config.DOMAIN_FOOD
        clf.bt_model = "arcface"
        clf.cache_dir = self.test_dir
        clf._trained = True
        clf.food_flavors_dict = {}
        clf.bt_to_gk_umbrella = {}
        clf.bt_gk_map = {}

        clf.batch_predict_bt = MagicMock(return_value=[("Pizza", 0.92, "trained", [])])
        clf.batch_predict_third_tag = MagicMock(return_value=[("Italian", 0.88, "trained")])
        clf.batch_predict_gk = MagicMock(return_value=[(["Cheese", "Crust"], 0.85, "trained")])
        clf.get_guaranteed_gk = MagicMock(return_value=[])

        mock_vs = MagicMock()
        mock_vs.hybrid_search.return_value = []

        query_embeddings = {"dense": np.random.randn(1, 1024).astype(np.float32), "sparse": [{}]}
        res = tag_all_skus(
            sku_names=["Margherita Pizza"],
            sku_categories=["Food"],
            query_embeddings=query_embeddings,
            vector_store=mock_vs,
            reranker=None,
            classifier=clf,
            sku_descriptions=["Classic cheese pizza"],
            sku_prices=[12.99],
        )

        self.assertEqual(len(res), 1)
        item = res[0]
        self.assertEqual(item["suggested_bt"], "Pizza")
        self.assertEqual(item["model"], "arcface")
        self.assertEqual(item["bt_model"], "arcface")
        self.assertIn("Classifier (arcface):", item["reasoning"])

    @patch("scripts.ml.train_bt_head.extract_embeddings")
    def test_train_bt_head_synthetic_pipeline(self, mock_extract):
        """Test train_bt_arcface end-to-end with mock catalog and embeddings."""
        from scripts.ml.train_bt_head import train_bt_arcface

        # Mock embeddings to avoid loading heavy 2.2GB BGE-M3 model in tests
        mock_extract.return_value = np.random.randn(20, 1024).astype(np.float32)

        # Create synthetic catalog CSV
        sample_df = pd.DataFrame(
            {
                "Name": [f"Item {i}" for i in range(20)],
                "basictype": [f"BT_{i % 3}" for i in range(20)],
                "Description": [f"Desc {i}" for i in range(20)],
                "Category": [f"Cat {i % 2}" for i in range(20)],
                "Price": [10.0 + i for i in range(20)],
            }
        )
        sample_path = os.path.join(self.test_dir, "sample.xlsx")
        with pd.ExcelWriter(sample_path) as writer:
            sample_df.to_excel(writer, sheet_name=config.FOOD_CATALOG_SHEET, index=False)

        artifacts = train_bt_arcface(
            domain=config.DOMAIN_FOOD,
            epochs=2,
            batch_size=8,
            lr=1e-3,
            device_name="cpu",
            from_sample=True,
            sample_file=sample_path,
            no_quantize=False,
            output_dir=self.test_dir,
        )

        self.assertTrue(os.path.exists(artifacts["fp32_onnx"]))
        self.assertTrue(os.path.exists(artifacts["int8_onnx"]))
        self.assertTrue(os.path.exists(artifacts["labels_json"]))
        self.assertTrue(os.path.exists(artifacts["meta_joblib"]))

        # Verify labels JSON contents
        with open(artifacts["labels_json"], encoding="utf-8") as f:
            labels_data = json.load(f)
        self.assertEqual(labels_data["domain"], config.DOMAIN_FOOD)
        self.assertEqual(labels_data["num_classes"], 3)
        self.assertEqual(len(labels_data["classes"]), 3)
        self.assertIn("price_scaler", labels_data)


if __name__ == "__main__":
    unittest.main()
