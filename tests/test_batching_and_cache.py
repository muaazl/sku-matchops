"""ONNX/Qdrant micro-batching with order preservation, Redis-backed rule caching, evaluator."""

import os
import time
import uuid
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import text

from engine import config
from engine.classification.classifier import ZeroShotClassifier
from engine.data_pipeline.vector_store import VectorStore
from engine.nlp.embedding_engine import EmbeddingEngine
from engine.rules_engine import evaluator, loader
from engine.rules_engine.evaluator import evaluate_conditions
from engine.rules_engine.loader import Rule

SIZES = [0, 1, 31, 32, 33, 100]  # empty, single, around the boundary, non-multiple of 32


# ---------------------------------------------------------------- ONNX micro-batching


class FakeSession:
    """ONNX session stand-in that records the batch size of every run() call."""

    def __init__(self, fn):
        self.fn = fn
        self.batch_sizes = []

    def run(self, _outputs, inputs):
        arr = next(iter(inputs.values()))
        self.batch_sizes.append(len(arr))
        return self.fn(inputs)


def _expected_batches(n: int, bs: int) -> list[int]:
    return [min(bs, n - i) for i in range(0, n, bs)]


@pytest.fixture
def onnx_batch_32(monkeypatch):
    monkeypatch.setattr(config, "ONNX_BATCH_SIZE", 32)
    return 32


@pytest.mark.parametrize("n", SIZES)
def test_arcface_bt_inference_is_micro_batched_and_ordered(onnx_batch_32, n):
    n_classes = 128
    clf = ZeroShotClassifier.__new__(ZeroShotClassifier)
    clf.bt_model = "arcface"
    clf._arcface_input_name = "x"
    clf._arcface_classes = [f"class_{i}" for i in range(n_classes)]
    clf._arcface_price_scaler = None
    clf.cold_start_router = None

    def logits(inputs):
        # Row i is tagged via feature 0 -> class i (very confident), so order is checkable
        x = inputs["x"]
        out = np.zeros((len(x), n_classes), dtype=np.float32)
        out[np.arange(len(x)), x[:, 0].astype(int)] = 50.0
        return [out]

    clf._arcface_session = FakeSession(logits)
    vecs = np.zeros((n, 4), dtype=np.float32)
    vecs[:, 0] = np.arange(n)

    preds = clf.batch_predict_bt(vecs, prices=[0.0] * n)

    assert [p[0] for p in preds] == [f"class_{i}" for i in range(n)]
    assert clf._arcface_session.batch_sizes == _expected_batches(n, 32)


class FakeTokenizer:
    """Encodes each text as one token id = its integer value, so outputs can be traced."""

    def __call__(self, texts, texts2=None, **_kw):
        ids = np.array([[int(t)] for t in texts], dtype=np.int64)
        return {"input_ids": ids, "attention_mask": np.ones_like(ids)}


@pytest.mark.parametrize("n", SIZES)
def test_bi_encoder_is_micro_batched_and_ordered(onnx_batch_32, n):
    eng = EmbeddingEngine.__new__(EmbeddingEngine)
    eng.tokenizer = FakeTokenizer()

    def hybrid(inputs):
        ids = inputs["input_ids"][:, 0].astype(np.float32)
        dense = np.stack([ids + 1, np.ones_like(ids)], axis=1)
        return [dense, np.ones((len(ids), 1), dtype=np.float32)]

    eng.bi_session = FakeSession(hybrid)
    texts = [str(i) for i in range(n)]

    out = eng.encode(texts)

    if n == 0:
        assert len(out["dense"]) == 0 and out["sparse"] == []
    else:
        assert [next(iter(s)) for s in out["sparse"]] == list(range(n))
        expected = np.array([[i + 1, 1] for i in range(n)], dtype=np.float32)
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        np.testing.assert_allclose(out["dense"], expected, rtol=1e-6)
    assert eng.bi_session.batch_sizes == _expected_batches(n, 32)


@pytest.mark.parametrize("n", SIZES)
def test_cross_encoder_is_micro_batched_and_ordered(onnx_batch_32, n):
    eng = EmbeddingEngine.__new__(EmbeddingEngine)
    eng.rerank_tokenizer = FakeTokenizer()
    eng.cross_session = FakeSession(lambda inp: [inp["input_ids"].astype(np.float32)])
    pairs = [[str(i), "x"] for i in range(n)]

    scores = eng.score_cross_encoder(pairs)

    assert scores.tolist() == [float(i) for i in range(n)]
    assert eng.cross_session.batch_sizes == _expected_batches(n, 32)


def test_classifier_embed_cache_keeps_one_file_per_input_set(tmp_path):
    calls = []
    clf = ZeroShotClassifier.__new__(ZeroShotClassifier)
    clf.cache_dir = str(tmp_path)
    clf.model = SimpleNamespace(
        encode=lambda texts: calls.append(list(texts)) or {"dense": np.ones((len(texts), 2))}
    )

    clf._embed_cached(["a", "b"], "food_bt")
    clf._embed_cached(["c"], "food_bt")  # e.g. the test DB's dictionaries
    clf._embed_cached(["a", "b"], "food_bt")  # the app's set must still be cached

    assert calls == [["a", "b"], ["c"]]
    assert len(list(tmp_path.glob("food_bt_*.pkl"))) == 2


def test_onnx_batch_size_is_env_configurable():
    import subprocess
    import sys

    code = "from engine import config; print(config.ONNX_BATCH_SIZE)"
    env = {**os.environ, "ONNX_BATCH_SIZE": "64"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert out.stdout.strip().splitlines()[-1] == "64"


# ---------------------------------------------------------------- Qdrant batch search


class FakeQdrant:
    """Echoes each request's query vector back as a hit, recording chunk sizes."""

    def __init__(self):
        self.chunk_sizes = []

    def query_batch_points(self, collection_name, requests):
        self.chunk_sizes.append(len(requests))
        out = []
        for req in requests:
            q = req.query if isinstance(req.query, list) else req.prefetch[0].query
            out.append(SimpleNamespace(points=[SimpleNamespace(payload={"q": q[0]}, score=1.0)]))
        return out


def _store(monkeypatch, chunk: int) -> VectorStore:
    monkeypatch.setattr(config, "QDRANT_BATCH_CHUNK_SIZE", chunk)
    vs = VectorStore.__new__(VectorStore)
    vs.client = FakeQdrant()
    vs._checked_collections = {config.QDRANT_COLLECTION_MARKET, config.QDRANT_COLLECTION_FOOD}
    return vs


@pytest.mark.parametrize("n", SIZES)
def test_qdrant_hybrid_batch_search_chunks_and_preserves_order(monkeypatch, n):
    vs = _store(monkeypatch, chunk=32)
    dense = [np.array([float(i), 0.0]) for i in range(n)]

    hits = vs.search_batch_from_vectors(dense, [{}] * n, [None] * n, domain="market", top_k=5)

    assert [h[0]["q"] for h in hits] == [float(i) for i in range(n)]
    assert vs.client.chunk_sizes == _expected_batches(n, 32)


@pytest.mark.parametrize("n", SIZES)
def test_qdrant_neighbor_batch_search_uses_configured_chunk(monkeypatch, n):
    vs = _store(monkeypatch, chunk=32)
    dense = [np.array([float(i), 0.0]) for i in range(n)]

    hits = vs.search_batch_catalog_neighbors(dense, domain="market", top_k=5)

    assert [h[0]["q"] for h in hits] == [float(i) for i in range(n)]
    assert vs.client.chunk_sizes == _expected_batches(n, 32)


def test_qdrant_neighbor_failure_keeps_alignment(monkeypatch):
    vs = _store(monkeypatch, chunk=32)
    real = vs.client.query_batch_points

    def flaky(collection_name, requests):
        if vs.client.chunk_sizes:  # second chunk fails
            raise ValueError("boom")
        return real(collection_name, requests)

    vs.client.query_batch_points = flaky
    hits = vs.search_batch_catalog_neighbors(
        [np.array([float(i), 0.0]) for i in range(100)], domain="market"
    )
    assert len(hits) == 100  # one slot per input SKU, never shifted
    assert [h[0]["q"] for h in hits[:32]] == [float(i) for i in range(32)]
    assert all(h == [] for h in hits[32:])


# ---------------------------------------------------------------- Rule caching + Redis


class FakeRedis:
    def __init__(self):
        self.kv, self.published = {}, []

    def get(self, key):
        return self.kv.get(key)

    def set(self, key, value, nx=False):
        if nx and key in self.kv:
            return False
        self.kv[key] = value
        return True

    def publish(self, channel, msg):
        self.published.append((channel, msg))
        return 1


class DownRedis:
    def __getattr__(self, _name):
        def fail(*_a, **_kw):
            raise ConnectionError("redis down")

        return fail


@pytest.fixture
def rule_db(pg, monkeypatch):
    """A test rule in PostgreSQL, a fresh in-memory cache, and a DB-load counter."""
    rid = f"t{uuid.uuid4().hex[:10]}"
    with pg.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO rules (rule_id, domain, priority, description, reasoning) "
                "VALUES (:r, 'market', 5, 'test', 'test')"
            ),
            {"r": rid},
        )
        conn.execute(
            text(
                "INSERT INTO conditions (rule_id, condition_type, value) VALUES (:r, 'sku_contains', 'zzz')"
            ),
            {"r": rid},
        )

    loads = []
    real_load = loader.load_rules_from_db

    def counting_load():
        loads.append(1)
        real_load()

    monkeypatch.setattr(loader, "load_rules_from_db", counting_load)
    monkeypatch.setattr(loader, "_RULES_CACHE", None)
    loader._cached_rules.cache_clear()

    yield SimpleNamespace(rule_id=rid, loads=loads, engine=pg)

    with pg.begin() as conn:
        conn.execute(text("DELETE FROM rules WHERE rule_id = :r"), {"r": rid})
    loader.refresh_rules_cache()


def _set_priority(pg, rule_id: str, priority: int):
    with pg.begin() as conn:
        conn.execute(
            text("UPDATE rules SET priority = :p WHERE rule_id = :r"), {"p": priority, "r": rule_id}
        )


def test_rules_load_once_and_cache_hits_skip_db(rule_db, monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(loader, "_redis", lambda: fake)

    for _ in range(50):  # many SKU evaluations
        assert any(r.rule_id == rule_db.rule_id for r in loader.get_rules("market"))
    v1 = loader.sync_rules_version()
    for _ in range(5):  # many jobs, nothing changed
        assert loader.sync_rules_version() == v1

    assert len(rule_db.loads) == 1
    assert fake.kv[loader.RULES_VERSION_KEY] == v1  # empty Redis is seeded


def test_rule_change_bumps_version_publishes_and_engine_reloads(rule_db, monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(loader, "_redis", lambda: fake)
    v1 = loader.sync_rules_version()

    # Gateway: rule updated + invalidation published (this is a separate process in prod)
    _set_priority(rule_db.engine, rule_db.rule_id, 7)
    assert loader.publish_rules_invalidation() is True
    v2 = fake.kv[loader.RULES_VERSION_KEY]
    assert v2 != v1
    assert fake.published == [(loader.RULES_INVALIDATE_CHANNEL, v2)]

    # Engine whose cache is stale picks it up at the next job start
    monkeypatch.setattr(loader, "_RULES_VERSION", v1)
    loads_before = len(rule_db.loads)
    assert loader.sync_rules_version() == v2
    assert len(rule_db.loads) == loads_before + 1
    rule = next(r for r in loader.get_rules("market") if r.rule_id == rule_db.rule_id)
    assert rule.priority == 7


def test_redis_down_falls_back_to_postgres(rule_db, monkeypatch):
    monkeypatch.setattr(loader, "_redis", lambda: DownRedis())
    loader.sync_rules_version()

    _set_priority(rule_db.engine, rule_db.rule_id, 9)
    assert loader.publish_rules_invalidation() is False  # gateway then uses the HTTP fallback

    loads_before = len(rule_db.loads)
    loader.sync_rules_version()  # engine reloads straight from PostgreSQL
    assert len(rule_db.loads) == loads_before + 1
    rule = next(r for r in loader.get_rules("market") if r.rule_id == rule_db.rule_id)
    assert rule.priority == 9


def test_gateway_falls_back_to_http_refresh_when_redis_down(monkeypatch):
    from backend.app.services import rules_service

    calls = []
    monkeypatch.setattr(rules_service, "publish_rules_invalidation", lambda: False)
    monkeypatch.setattr(rules_service, "refresh_engine_rules", lambda: calls.append("http"))
    rules_service._refresh_rules_everywhere()
    assert calls == ["http"]

    calls.clear()
    monkeypatch.setattr(rules_service, "publish_rules_invalidation", lambda: True)
    rules_service._refresh_rules_everywhere()
    assert calls == []


def test_failed_reload_keeps_previous_rules(rule_db, monkeypatch):
    loader.get_rules("market")
    version = loader.get_rules_version()

    def broken():
        raise ConnectionError("db down")

    monkeypatch.setattr(loader, "get_connection", broken)
    loader.refresh_rules_cache()
    assert loader.get_rules_version() == version
    assert any(r.rule_id == rule_db.rule_id for r in loader.get_rules("market"))


def test_pinned_snapshot_prevents_mid_job_rule_drift(rule_db, monkeypatch):
    """A rule reload while a job is in flight must not change that job's rules or rule_version."""
    from engine.rules_engine import run_rules_engine

    monkeypatch.setattr(loader, "_redis", lambda: FakeRedis())
    with rule_db.engine.begin() as conn:
        conn.execute(
            text("UPDATE conditions SET value = 'anchor' WHERE rule_id = :r"),
            {"r": rule_db.rule_id},
        )
        conn.execute(
            text(
                "INSERT INTO actions (rule_id, action_type, value) VALUES (:r, 'add_gk', 'OldTag')"
            ),
            {"r": rule_db.rule_id},
        )
    loader.refresh_rules_cache()

    def run_chunk(n):
        return [
            run_rules_engine({"domain": "market", "sku_name": f"Anchor {i}", "gk": []})["gk"]
            for i in range(n)
        ]

    # Worker: sync + snapshot at job start, then process chunks under the pin
    loader.sync_rules_version()
    snapshot = loader.snapshot_rules()
    with loader.pinned_rules(snapshot):
        chunk1 = run_chunk(3)

        # Rule edited and broadcast mid-job; the listener reloads the live cache
        with rule_db.engine.begin() as conn:
            conn.execute(
                text("UPDATE actions SET value = 'NewTag' WHERE rule_id = :r"),
                {"r": rule_db.rule_id},
            )
        loader.refresh_rules_cache()
        assert loader.get_rules_version() != snapshot.version  # live cache moved on

        chunk2 = run_chunk(3)

    assert chunk1 == chunk2 == [["OldTag"]] * 3  # no drift inside the job
    assert snapshot.version != loader.get_rules_version()
    # The next job (no pin) sees the new rules immediately
    assert run_chunk(1) == [["NewTag"]]


def test_snapshot_is_consistent_pair_and_pin_is_scoped(rule_db):
    snapshot = loader.snapshot_rules()
    assert snapshot.version == loader.get_rules_version()
    with loader.pinned_rules(snapshot):
        inside = loader.get_rules("market")
    assert isinstance(inside, tuple)  # immutable view
    assert loader._PINNED.get() is None  # pin released after the job


REDIS_URL = os.getenv("TEST_REDIS_URL", "redis://localhost:6379/15")


def test_pubsub_listener_reloads_immediately(rule_db, monkeypatch):
    import redis

    try:
        redis.Redis.from_url(REDIS_URL, socket_connect_timeout=1).ping()
    except Exception:
        pytest.skip(f"Redis not available at {REDIS_URL}")

    monkeypatch.setattr(config, "REDIS_URL", REDIS_URL)
    monkeypatch.setattr(loader, "_redis_client", None)
    loader.get_rules("market")
    loader.start_invalidation_listener()

    _set_priority(rule_db.engine, rule_db.rule_id, 11)
    publisher = redis.Redis.from_url(REDIS_URL)

    # Publish the way the gateway does, but without reloading this process first
    deadline = time.time() + 5
    while time.time() < deadline:
        publisher.publish(loader.RULES_INVALIDATE_CHANNEL, "new-version")
        time.sleep(0.2)
        rule = next(r for r in loader.get_rules("market") if r.rule_id == rule_db.rule_id)
        if rule.priority == 11:
            break
    assert rule.priority == 11


# ---------------------------------------------------------------- evaluator


def _rule(conditions, logic="AND"):
    r = Rule((0, "r1", "market", 10, "d", "why", logic, 1))
    r.conditions = conditions
    return r


def _c(ctype, value, group=1, negate=False):
    return {"condition_group": group, "condition_type": ctype, "value": value, "negate": negate}


@pytest.fixture
def flavors(monkeypatch):
    monkeypatch.setattr(
        evaluator,
        "_FLAVOR_CACHE",
        {
            "flavors_dict": {
                "chicken": "chicken",
                "chk": "chicken",
                "prawn": "prawn",
                "veg": "veg",
            },
            "meat_flavors": {"chicken", "prawn"},
            "vegetable_flavors": {"veg"},
            "seafood_flavors": {"prawn"},
        },
    )
    evaluator._extract_flavors_from_sku_cached.cache_clear()
    yield
    evaluator._extract_flavors_from_sku_cached.cache_clear()


RECORD = {
    "sku_name": "Anchor Butter 200g",
    "bt": "Butter",
    "gk": ["Dairy", "Spread"],
    "category": "Dairy & Eggs",
    "region": "Western",
    "price": 850.0,
}


@pytest.mark.parametrize(
    "cond, expected",
    [
        (_c("sku_contains", "BUTTER"), True),
        (_c("sku_contains", "cheese"), False),
        (_c("bt_is", "butter"), True),
        (_c("bt_is", "butt"), False),  # exact match, not substring
        (_c("gk_contains", "dairy"), True),
        (_c("gk_contains", "dai"), False),  # list membership, not substring
        (_c("category_contains", "eggs"), True),
        (_c("region_is", "western"), True),
        (_c("region_is", "west"), False),
        (_c("sku_contains", "butter", negate=True), False),
        (_c("sku_contains", "cheese", negate=True), True),
        (_c("unknown_type", "x"), False),
    ],
)
def test_single_condition_matching(cond, expected):
    assert evaluate_conditions(_rule([cond]), RECORD) is expected


@pytest.mark.parametrize(
    "ctype, threshold, expected",
    [
        ("price_below", "850", False),  # strict cutoff: equal is not below
        ("price_below", "850.01", True),
        ("price_above", "850", False),  # strict cutoff: equal is not above
        ("price_above", "849.99", True),
        ("price_below", "abc", False),  # malformed threshold -> condition is false, no crash
    ],
)
def test_price_threshold_cutoffs(ctype, threshold, expected):
    assert evaluate_conditions(_rule([_c(ctype, threshold)]), RECORD) is expected


def test_no_conditions_always_fires():
    assert evaluate_conditions(_rule([]), {}) is True


def test_same_type_in_group_is_or_different_types_are_and():
    rule = _rule([_c("bt_is", "cheese"), _c("bt_is", "butter"), _c("sku_contains", "anchor")])
    assert evaluate_conditions(rule, RECORD) is True
    rule = _rule([_c("bt_is", "butter"), _c("sku_contains", "cheese")])
    assert evaluate_conditions(rule, RECORD) is False


@pytest.mark.parametrize("logic, expected", [("AND", False), ("OR", True), ("or", True)])
def test_condition_logic_across_groups(logic, expected):
    rule = _rule([_c("bt_is", "butter", group=1), _c("bt_is", "cheese", group=2)], logic=logic)
    assert evaluate_conditions(rule, RECORD) is expected


@pytest.mark.parametrize(
    "record",
    [
        {},  # empty record
        {"sku_name": None, "bt": None, "gk": [], "category": None, "region": None, "price": None},
        {"sku_name": 12345, "bt": 7, "gk": [None, 3], "price": "not-a-number"},
    ],
)
def test_malformed_records_never_raise(record):
    rule = _rule(
        [
            _c("sku_contains", "x", group=1),
            _c("bt_is", "x", group=2),
            _c("gk_contains", "x", group=3),
            _c("category_contains", "x", group=4),
            _c("region_is", "x", group=5),
            _c("price_below", "10", group=6),
            _c("price_above", "10", group=7),
        ],
        logic="OR",
    )
    assert evaluate_conditions(rule, record) is False


def test_missing_condition_value_is_treated_as_false():
    rule = _rule([{"condition_group": 1, "condition_type": "sku_contains"}])  # no "value"
    assert evaluate_conditions(rule, RECORD) is False


@pytest.mark.parametrize(
    "sku, cond, expected",
    [
        ("Chicken Fried Rice", _c("flavor_contains", "chicken"), True),
        ("Chk Kottu", _c("flavor_contains", "chicken"), True),  # alias -> canonical
        ("Chk Kottu", _c("flavor_contains", "chk"), True),  # target alias canonicalised
        ("Chickenfried Rice", _c("flavor_contains", "chicken"), False),  # word boundary
        ("Chicken Fried Rice", _c("flavor_is", "meat"), True),
        ("Prawn Fried Rice", _c("flavor_is", "meat"), False),  # seafood excluded from meat
        ("Prawn Fried Rice", _c("flavor_is", "seafood"), True),
        ("Veg Fried Rice", _c("flavor_is", "veg"), True),
        ("Veg Fried Rice", _c("flavor_is", "dessert"), False),  # unknown category
        ("", _c("flavor_contains", "chicken"), False),
    ],
)
def test_flavor_conditions(flavors, sku, cond, expected):
    assert evaluate_conditions(_rule([cond]), {"sku_name": sku}) is expected
