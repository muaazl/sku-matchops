"""ONNX/Qdrant micro-batching with order preservation, Redis-backed rule caching, evaluator."""

import json
import os
import time
import uuid
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import text

from backend.services import rule_service
from engine.core import config
from engine.inference.embedding_model import EmbeddingModel
from engine.inference.gliner_model import GlinerModel
from engine.inference.reranker_model import RerankerModel
from engine.pipelines import rules_evaluation_pipeline as evaluator
from engine.pipelines import rules_loader_pipeline as loader
from engine.pipelines.classification_pipeline import TaxonomyClassifier
from engine.pipelines.rules_evaluation_pipeline import evaluate_conditions
from engine.pipelines.rules_loader_pipeline import Rule
from engine.vector_store.vector_store import VectorStore

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
    clf = TaxonomyClassifier.__new__(TaxonomyClassifier)
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

    all_special_ids = []

    def __call__(self, texts, texts2=None, **_kw):
        ids = np.array([[int(t)] for t in texts], dtype=np.int64)
        return {"input_ids": ids, "attention_mask": np.ones_like(ids)}


@pytest.mark.parametrize("n", SIZES)
def test_bi_encoder_is_micro_batched_and_ordered(onnx_batch_32, n):
    eng = EmbeddingModel.__new__(EmbeddingModel)
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
    eng = RerankerModel.__new__(RerankerModel)
    eng.rerank_tokenizer = FakeTokenizer()
    eng.cross_session = FakeSession(lambda inp: [inp["input_ids"].astype(np.float32)])
    pairs = [[str(i), "x"] for i in range(n)]

    scores = eng.score_cross_encoder(pairs)

    assert scores.tolist() == [float(i) for i in range(n)]
    assert eng.cross_session.batch_sizes == _expected_batches(n, 32)


def test_classifier_embed_cache_keeps_one_file_per_input_set(tmp_path):
    calls = []
    clf = TaxonomyClassifier.__new__(TaxonomyClassifier)
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

    code = "from engine.core import config; print(config.ONNX_BATCH_SIZE)"
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

    hits = vs.search_catalog_hybrid_batch(dense, [{}] * n, [None] * n, domain="market", top_k=5)

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
    """A test rule in PostgreSQL, a fresh Engine in-memory cache, and a Redis-load counter."""
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
    real_load = loader.load_rules_from_cache

    def counting_load(*args, **kwargs):
        loads.append(1)
        real_load(*args, **kwargs)

    monkeypatch.setattr(loader, "load_rules_from_cache", counting_load)
    monkeypatch.setattr(loader, "_RULES_CACHE", None)
    loader._cached_rules.cache_clear()

    yield SimpleNamespace(rule_id=rid, loads=loads, engine=pg)

    with pg.begin() as conn:
        conn.execute(text("DELETE FROM rules WHERE rule_id = :r"), {"r": rid})
    monkeypatch.setattr(
        rule_service.engine_gateway_service, "refresh_engine_rules", lambda **kw: {"status": "ok"}
    )
    rule_service.sync_rules_everywhere()


def _set_priority(pg, rule_id: str, priority: int):
    with pg.begin() as conn:
        conn.execute(
            text("UPDATE rules SET priority = :p WHERE rule_id = :r"), {"p": priority, "r": rule_id}
        )


def test_rules_load_once_and_cache_hits_skip_redis_reload(rule_db, monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(loader, "_redis", lambda: fake)
    # Gateway publishes PostgreSQL rules snapshot into Redis
    rule_service.sync_rules_everywhere()

    # Simulate a cold Engine worker starting with empty memory and missing version key
    monkeypatch.setattr(loader, "_RULES_CACHE", None)
    loader._cached_rules.cache_clear()
    rule_db.loads.clear()
    fake.kv.pop(loader.RULES_VERSION_KEY, None)

    for _ in range(50):  # many SKU evaluations
        assert any(r.rule_id == rule_db.rule_id for r in loader.get_rules("market"))
    v1 = loader.sync_rules_version()
    for _ in range(5):  # many jobs, nothing changed
        assert loader.sync_rules_version() == v1

    assert len(rule_db.loads) == 1
    assert fake.kv[loader.RULES_VERSION_KEY] == v1  # missing version key is seeded


def test_rule_change_bumps_version_publishes_and_engine_reloads(rule_db, monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(loader, "_redis", lambda: fake)
    rule_service.sync_rules_everywhere()
    v1 = loader.sync_rules_version()
    old_cache = loader._RULES_CACHE
    fake.published.clear()

    # Gateway: rule updated in PostgreSQL + snapshot/invalidation published to Redis
    _set_priority(rule_db.engine, rule_db.rule_id, 7)
    v2, published = rule_service.sync_rules_everywhere()
    assert published is True
    assert fake.kv[loader.RULES_VERSION_KEY] == v2
    assert v2 != v1
    assert fake.published == [(loader.RULES_INVALIDATE_CHANNEL, v2)]

    # Separate Engine worker whose in-memory cache is still on v1 picks up v2 at next job start
    monkeypatch.setattr(loader, "_RULES_CACHE", old_cache)
    monkeypatch.setattr(loader, "_RULES_VERSION", v1)
    loader._cached_rules.cache_clear()

    loads_before = len(rule_db.loads)
    assert loader.sync_rules_version() == v2
    assert len(rule_db.loads) == loads_before + 1
    rule = next(r for r in loader.get_rules("market") if r.rule_id == rule_db.rule_id)
    assert rule.priority == 7


def test_redis_down_pushes_rules_via_grpc_refresh(rule_db, monkeypatch):
    monkeypatch.setattr(loader, "_redis", lambda: DownRedis())
    grpc_pushes = []

    def fake_grpc_refresh(rules_data=None, version=""):
        grpc_pushes.append((rules_data, version))
        loader.set_rules_from_dicts(rules_data or [], version=version or None)
        return {"status": "success"}

    monkeypatch.setattr(
        rule_service.engine_gateway_service, "refresh_engine_rules", fake_grpc_refresh
    )

    _set_priority(rule_db.engine, rule_db.rule_id, 9)
    v2, published = rule_service.sync_rules_everywhere()
    assert published is False
    assert len(grpc_pushes) == 1
    assert grpc_pushes[0][1] == v2

    # Engine keeps the gRPC-pushed rule snapshot when Redis remains unreachable
    assert loader.sync_rules_version() == v2
    rule = next(r for r in loader.get_rules("market") if r.rule_id == rule_db.rule_id)
    assert rule.priority == 9


def test_gateway_falls_back_to_grpc_refresh_when_redis_down(monkeypatch):
    calls = []
    monkeypatch.setattr(
        rule_service,
        "load_active_rules",
        lambda db=None: [{"rule_id": "1"}],
    )
    monkeypatch.setattr(
        rule_service,
        "publish_rules_snapshot",
        lambda data: ("v1", False),
    )
    monkeypatch.setattr(
        rule_service.engine_gateway_service,
        "refresh_engine_rules",
        lambda rules_data=None, version="": calls.append((rules_data, version)),
    )
    rule_service.sync_rules_everywhere()
    assert calls == [([{"rule_id": "1"}], "v1")]

    calls.clear()
    monkeypatch.setattr(
        rule_service,
        "publish_rules_snapshot",
        lambda data: ("v1", True),
    )
    rule_service.sync_rules_everywhere()
    assert calls == []


def test_failed_reload_keeps_previous_rules(rule_db, monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(loader, "_redis", lambda: fake)
    rule_service.sync_rules_everywhere()
    loader.get_rules("market")
    version = loader.get_rules_version()

    monkeypatch.setattr(loader, "_redis", lambda: DownRedis())
    loader.refresh_rules_cache()
    assert loader.get_rules_version() == version
    assert any(r.rule_id == rule_db.rule_id for r in loader.get_rules("market"))


def test_pinned_snapshot_prevents_mid_job_rule_drift(rule_db, monkeypatch):
    """A rule reload while a job is in flight must not change that job's rules or rule_version."""
    from engine.pipelines.rules_evaluation_pipeline import run_rules_engine

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
    rule_service.sync_rules_everywhere()

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

        # Rule edited and broadcast mid-job; live cache moves on
        with rule_db.engine.begin() as conn:
            conn.execute(
                text("UPDATE actions SET value = 'NewTag' WHERE rule_id = :r"),
                {"r": rule_db.rule_id},
            )
        rule_service.sync_rules_everywhere()
        assert loader.get_rules_version() != snapshot.version

        chunk2 = run_chunk(3)

    assert chunk1 == chunk2 == [["OldTag"]] * 3  # no drift inside the job
    assert snapshot.version != loader.get_rules_version()
    # The next job (no pin) sees the new rules immediately
    assert run_chunk(1) == [["NewTag"]]


def test_snapshot_is_consistent_pair_and_pin_is_scoped(rule_db, monkeypatch):
    monkeypatch.setattr(loader, "_redis", lambda: FakeRedis())
    rule_service.sync_rules_everywhere()
    snapshot = loader.snapshot_rules()
    assert snapshot.version == loader.get_rules_version()
    with loader.pinned_rules(snapshot):
        inside = loader.get_rules("market")
    assert isinstance(inside, tuple)  # immutable view
    assert loader._PINNED.get() is None  # pin released after the job


REDIS_URL = os.getenv("TEST_REDIS_URL", "redis://localhost:6379/15")


def test_pubsub_listener_reloads_immediately(rule_db, monkeypatch):
    import threading

    import redis

    try:
        redis.Redis.from_url(REDIS_URL, socket_connect_timeout=1).ping()
    except Exception:
        pytest.skip(f"Redis not available at {REDIS_URL}")

    monkeypatch.setattr(config, "REDIS_URL", REDIS_URL)
    monkeypatch.setattr(loader, "_redis_client", None)
    rule_service.sync_rules_everywhere()
    loader.get_rules("market")
    stop_event = threading.Event()
    listener_thread = loader.start_invalidation_listener(stop_event=stop_event)

    try:
        _set_priority(rule_db.engine, rule_db.rule_id, 11)
        updated_rules = rule_service.load_active_rules()
        publisher = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        publisher.set(loader.RULES_DATA_KEY, json.dumps(updated_rules, default=str))
        publisher.set(loader.RULES_VERSION_KEY, "new-version")

        # Publish the way the gateway does, without mutating this process's in-memory cache directly
        deadline = time.time() + 5
        rule = next(r for r in loader.get_rules("market") if r.rule_id == rule_db.rule_id)
        while time.time() < deadline:
            publisher.publish(loader.RULES_INVALIDATE_CHANNEL, "new-version")
            time.sleep(0.2)
            rule = next(r for r in loader.get_rules("market") if r.rule_id == rule_db.rule_id)
            if rule.priority == 11:
                break
        assert rule.priority == 11
    finally:
        stop_event.set()
        listener_thread.join(timeout=3)


def test_start_invalidation_listener_singleton_reuses_live_thread(monkeypatch):
    class _FakeThread:
        def __init__(self, target=None, name=None, daemon=None):
            self.name = name
            self._alive = True

        def start(self):
            return None

        def is_alive(self):
            return self._alive

    spawned = []

    def make_thread(*args, **kwargs):
        t = _FakeThread(*args, **kwargs)
        spawned.append(t)
        return t

    monkeypatch.setattr(loader.threading, "Thread", make_thread)
    monkeypatch.setattr(loader, "_invalidation_thread", None)
    try:
        t1 = loader.start_invalidation_listener()
        t2 = loader.start_invalidation_listener()
        assert t1 is t2
        assert len(spawned) == 1

        t1._alive = False
        t3 = loader.start_invalidation_listener()
        assert t3 is not t1
        assert len(spawned) == 2
    finally:
        loader._invalidation_thread = None


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
                "chicken breast": "chicken",
                "chk": "chicken",
                "prawn": "prawn",
                "veg": "veg",
                "spinach": "spinach",
            },
            "meat_flavors": {"chicken", "prawn"},
            "vegetable_flavors": {"veg", "spinach"},
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
        ("Tasty Chicken Breast Soup", _c("flavor_contains", "chicken"), True),  # multi-word alias
        ("Chk Kottu", _c("flavor_contains", "chicken"), True),  # alias -> canonical
        ("Chk Kottu", _c("flavor_contains", "chk"), True),  # target alias canonicalised
        ("Chickenfried Rice", _c("flavor_contains", "chicken"), False),  # word boundary
        ("Chicken Fried Rice", _c("flavor_is", "meat"), True),
        ("Prawn Fried Rice", _c("flavor_is", "meat"), False),  # seafood excluded from meat
        ("Prawn Fried Rice", _c("flavor_is", "seafood"), True),
        ("Veg Fried Rice", _c("flavor_is", "veg"), True),
        ("Spinach Noodles 500g", _c("flavor_is", "vegetable"), True),  # "vegetable" alias
        ("Spinach Noodles 500g", _c("flavor_contains", "spinach"), True),
        ("Veg Fried Rice", _c("flavor_is", "dessert"), False),  # unknown category
        ("", _c("flavor_contains", "chicken"), False),
    ],
)
def test_flavor_conditions(flavors, sku, cond, expected):
    assert evaluate_conditions(_rule([cond]), {"sku_name": sku}) is expected


def _op(ctype, value, operator="AND", negate=False):
    return {
        "condition_group": 1,
        "operator": operator,
        "condition_type": ctype,
        "value": value,
        "negate": negate,
    }


def test_per_condition_operators_and_precedence():
    from backend.services.rule_evaluation_service import evaluate_rule_conditions

    # (bt_is="Cheese" AND sku_contains="anchor") OR region_is="Western" -> False OR True -> True
    conds = [
        _op("bt_is", "cheese", operator="AND"),
        _op("sku_contains", "anchor", operator="AND"),
        _op("region_is", "western", operator="OR"),
    ]
    assert evaluate_conditions(_rule(conds), RECORD) is True
    assert evaluate_rule_conditions({"conditions": conds}, RECORD) is True

    # (sku_contains="anchor" AND bt_is="Butter") OR (sku_contains="milk" AND region_is="Central") -> True
    conds2 = [
        _op("sku_contains", "anchor", operator="AND"),
        _op("bt_is", "butter", operator="AND"),
        _op("sku_contains", "milk", operator="OR"),
        _op("region_is", "central", operator="AND"),
    ]
    assert evaluate_conditions(_rule(conds2), RECORD) is True
    assert evaluate_rule_conditions({"conditions": conds2}, RECORD) is True

    # (sku_contains="anchor" AND bt_is="Cheese") OR (sku_contains="anchor" AND NOT region_is="Western") -> False
    conds3 = [
        _op("sku_contains", "anchor", operator="AND"),
        _op("bt_is", "cheese", operator="AND"),
        _op("sku_contains", "anchor", operator="OR"),
        _op("region_is", "western", operator="AND", negate=True),
    ]
    assert evaluate_conditions(_rule(conds3), RECORD) is False
    assert evaluate_rule_conditions({"conditions": conds3}, RECORD) is False

    # Legacy DB rule migrated with default operator="AND", condition_logic="OR", condition_group=1,2
    legacy_or_conds = [
        {
            "condition_group": 1,
            "operator": "AND",
            "condition_type": "bt_is",
            "value": "cheese",
            "negate": 0,
        },
        {
            "condition_group": 2,
            "operator": "AND",
            "condition_type": "bt_is",
            "value": "butter",
            "negate": 0,
        },
    ]
    assert evaluate_conditions(_rule(legacy_or_conds, logic="OR"), RECORD) is True
    assert (
        evaluate_rule_conditions({"condition_logic": "OR", "conditions": legacy_or_conds}, RECORD)
        is True
    )


def test_per_condition_operator_same_type_uses_and():
    from backend.services.rule_evaluation_service import evaluate_rule_conditions

    # Same condition_type joined with explicit AND must require BOTH to match (not OR)
    conds_pass = [
        _op("sku_contains", "anchor", operator="AND"),
        _op("sku_contains", "butter", operator="AND"),
    ]
    conds_fail = [
        _op("sku_contains", "anchor", operator="AND"),
        _op("sku_contains", "cheese", operator="AND"),
    ]
    assert evaluate_conditions(_rule(conds_pass), RECORD) is True
    assert evaluate_rule_conditions({"conditions": conds_pass}, RECORD) is True
    assert evaluate_conditions(_rule(conds_fail), RECORD) is False
    assert evaluate_rule_conditions({"conditions": conds_fail}, RECORD) is False


def test_format_matcher_rerank_text_omits_empty_fields():
    from engine.preprocessing.text_normalizer import format_matcher_rerank_text

    full_market = format_matcher_rerank_text(
        name="Acme Almond Milk",
        seller_category="Dairy",
        brand_or_flavor="Acme",
        description="Unsweetened 1L",
        domain="market",
    )
    assert (
        full_market
        == "Name: Acme Almond Milk | SellerCategory: Dairy | Brand: Acme | Description: Unsweetened 1L"
    )

    partial_food = format_matcher_rerank_text(
        name="Chicken Fried Rice",
        seller_category="",
        brand_or_flavor="Chicken",
        description="nan",
        domain="food",
    )
    assert partial_food == "Name: Chicken Fried Rice | Flavor: Chicken"


def test_bm25_tf_and_last_token_dense_only_encode():
    from engine.preprocessing.bm25_tokenizer import compute_bm25_tf_weights

    tf1 = compute_bm25_tf_weights("Red Bull Energy Drink 250ml")
    tf2 = compute_bm25_tf_weights("Red Bull Energy Drink 250ml")
    assert tf1 and tf1 == tf2
    assert all(isinstance(k, int) and k > 0 and v > 0 for k, v in tf1.items())

    eng = EmbeddingModel.__new__(EmbeddingModel)
    eng.tokenizer = FakeTokenizer()
    eng._emb_pooling = "last_token"

    def last_token_hidden(inputs):
        ids = inputs["input_ids"]
        bsz = len(ids)
        hidden = np.zeros((bsz, 1, 4), dtype=np.float32)
        hidden[:, 0, 0] = 3.0
        hidden[:, 0, 1] = 4.0
        return [hidden]

    eng.bi_session = FakeSession(last_token_hidden)
    out = eng.encode(["101", "202"])
    assert out["dense"].shape == (2, 4)
    np.testing.assert_allclose(out["dense"][0], [0.6, 0.8, 0.0, 0.0], atol=1e-6)
    assert len(out["bm25"]) == 2 and all(isinstance(s, dict) and len(s) > 0 for s in out["bm25"])
    assert out["sparse"] == []


def test_listwise_lbnl_reranker_groups_by_query_and_extracts_cosine():
    class ListwiseTokenizer:
        def encode(self, text):
            return [1]

        def __call__(self, prompts, **_kw):
            prompt = prompts[0]
            ids = []
            for tok in prompt.replace("\n", " ").split(" "):
                if "<|embed_token|>" in tok:
                    ids.extend([10, 151670])
                elif "<|rerank_token|>" in tok:
                    ids.extend([10, 151671])
                elif tok:
                    ids.append(10)
            arr = np.array([ids], dtype=np.int64)
            return {"input_ids": arr, "attention_mask": np.ones_like(arr)}

    eng = RerankerModel.__new__(RerankerModel)
    eng._cross_arch = "listwise_lbnl"
    eng._score_activation = "identity"
    eng._doc_embed_token = "<|embed_token|>"
    eng._query_embed_token = "<|rerank_token|>"
    eng._doc_embed_token_id = 151670
    eng._query_embed_token_id = 151671
    eng.rerank_tokenizer = ListwiseTokenizer()

    def lbnl_proj(inputs):
        ids = inputs["input_ids"][0]
        seq_len = len(ids)
        emb = np.zeros((1, seq_len, 2), dtype=np.float32)
        doc_positions = np.where(ids == 151670)[0]
        q_positions = np.where(ids == 151671)[0]
        for idx, pos in enumerate(doc_positions):
            emb[0, pos] = [1.0, 0.0] if idx == 0 else [0.0, 1.0]
        emb[0, q_positions[-1]] = [1.0, 0.0]
        return [emb]

    eng.cross_session = FakeSession(lbnl_proj)
    pairs = [["Query A", "Doc 1"], ["Query A", "Doc 2"], ["Query B", "Doc 3"]]
    scores = eng.score_cross_encoder(pairs, batch_size=30)

    assert eng.cross_session.batch_sizes == [1, 1]
    np.testing.assert_allclose(scores, [1.0, 0.0, 1.0], atol=1e-6)


def test_config_folder_slug_and_strict_onnx_fail_fast(monkeypatch, tmp_path):
    import pytest as _pytest

    assert config.normalize_model_folder_name("BAAI/bge-m3") == "bge-m3"
    assert (
        config.normalize_model_folder_name("sentence-transformers/all-mpnet-base-v2")
        == "all-mpnet-base-v2"
    )
    assert config.normalize_model_folder_name("jinaai/jina-reranker-v3.5") == "jina-reranker-v3.5"

    missing = str(tmp_path / "missing_int8.onnx")
    monkeypatch.setattr(config, "EMBEDDING_ONNX", missing)
    with _pytest.raises(FileNotFoundError, match="FAIL-FAST"):
        EmbeddingModel()

    import pandas as pd

    monkeypatch.setattr(config, "GLINER_ONNX", missing)
    with _pytest.raises(FileNotFoundError, match="FAIL-FAST"):
        GlinerModel(pd.DataFrame(), domain="market")


def test_embedding_and_reranker_onnx_wrappers_cover_all_pooling_and_archs():
    import torch

    from scripts.sync import EmbeddingONNXWrapper, RerankerONNXWrapper

    class _DummyEnc(torch.nn.Module):
        def forward(self, input_ids, attention_mask, return_dict=True):
            bsz, seq = input_ids.shape
            hidden = torch.zeros((bsz, seq, 2), dtype=torch.float32)
            hidden[:, 0, :] = torch.tensor([1.0, 0.0])  # CLS token
            hidden[:, 1, :] = torch.tensor([0.0, 3.0])  # Real token 2
            hidden[:, 2, :] = torch.tensor([99.0, 99.0])  # Padded token (masked out)
            return type("Out", (), {"last_hidden_state": hidden})()

    ids = torch.tensor([[10, 20, 0]], dtype=torch.long)
    mask = torch.tensor([[1, 1, 0]], dtype=torch.long)

    out_cls = EmbeddingONNXWrapper(_DummyEnc(), pooling="cls")(ids, mask)
    np.testing.assert_allclose(out_cls.numpy(), [[1.0, 0.0]], atol=1e-6)

    out_mean = EmbeddingONNXWrapper(_DummyEnc(), pooling="mean")(ids, mask)
    np.testing.assert_allclose(out_mean.numpy(), [[0.5, 1.5]], atol=1e-6)

    out_last = EmbeddingONNXWrapper(_DummyEnc(), pooling="last_token")(ids, mask)
    np.testing.assert_allclose(out_last.numpy(), [[0.0, 3.0]], atol=1e-6)

    sparse_head = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        sparse_head.weight.copy_(torch.tensor([[1.0, 2.0]]))
    out_d, out_s = EmbeddingONNXWrapper(_DummyEnc(), pooling="cls", sparse_linear=sparse_head)(
        ids, mask
    )
    assert out_d.shape == (1, 2)
    assert out_s.shape == (1, 3)

    class _DummyCE(torch.nn.Module):
        def forward(self, input_ids, attention_mask, return_dict=True):
            return type("Out", (), {"logits": torch.tensor([[0.0], [2.0]])})()

    logits = RerankerONNXWrapper(_DummyCE(), arch_type="cross_encoder")(ids[:1], mask[:1])
    assert logits.shape == (2, 1)


def test_unified_sync_cli_parser_and_stage_modes():
    from scripts.sync import build_parser, resolve_stage_modes

    parser = build_parser()

    assert resolve_stage_modes(parser.parse_args([])) == {
        "models": "sync",
        "db": "sync",
        "cache": "sync",
        "qdrant": "sync",
        "arcface": "sync",
    }
    assert resolve_stage_modes(parser.parse_args(["--rebuild"])) == {
        "models": "rebuild",
        "db": "rebuild",
        "cache": "rebuild",
        "qdrant": "rebuild",
        "arcface": "rebuild",
    }
    assert resolve_stage_modes(parser.parse_args(["--sample"])) == {
        "models": "rebuild",
        "db": "rebuild",
        "cache": "rebuild",
        "qdrant": "rebuild",
        "arcface": "rebuild",
    }

    args_emb = parser.parse_args(["--models", "rebuild", "--model-target", "embedding"])
    assert args_emb.model_target == "embedding"
    assert resolve_stage_modes(args_emb) == {
        "models": "rebuild",
        "db": None,
        "cache": None,
        "qdrant": None,
        "arcface": None,
    }

    args_mixed = parser.parse_args(["--models", "export", "--rebuild"])
    assert resolve_stage_modes(args_mixed) == {
        "models": "export",
        "db": "rebuild",
        "cache": "rebuild",
        "qdrant": "rebuild",
        "arcface": "rebuild",
    }

    args_arc = parser.parse_args(["--arcface", "rebuild", "--epochs", "7"])
    assert args_arc.epochs == 7
    assert resolve_stage_modes(args_arc) == {
        "models": None,
        "db": None,
        "cache": None,
        "qdrant": None,
        "arcface": "rebuild",
    }


def test_refresh_model_config_updates_paths_and_dim(tmp_path, monkeypatch):
    import json

    from engine.core import config

    model_dir = tmp_path / "all-mpnet-base-v2"
    model_dir.mkdir()
    (model_dir / "model_manifest.json").write_text(
        json.dumps({"embedding_dim": 768, "pooling": "mean", "has_learned_sparse": False}),
        encoding="utf-8",
    )

    monkeypatch.setattr(config, "EMBEDDING_ROOT_DIR", str(tmp_path))
    monkeypatch.setenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
    monkeypatch.setenv("EMBEDDING_POOLING", "mean")
    try:
        config.refresh_model_config()
        assert config.EMBEDDING_MODEL == "sentence-transformers/all-mpnet-base-v2"
        assert config.EMBEDDING_DIM == 768
        assert config.EMBEDDING_ONNX == model_dir / "model_int8.onnx"
        assert config.check_learned_sparse_enabled() is False
    finally:
        monkeypatch.undo()
        config.refresh_model_config()


def test_thresholds_adapt_to_reranker_arch(tmp_path, monkeypatch):
    import json

    from engine.core import config

    lbnl_dir = tmp_path / "jina-reranker-v3.5"
    lbnl_dir.mkdir()
    (lbnl_dir / "model_manifest.json").write_text(
        json.dumps({"arch_type": "listwise_lbnl"}), encoding="utf-8"
    )

    ce_dir = tmp_path / "bge-reranker-v2-m3"
    ce_dir.mkdir()
    (ce_dir / "model_manifest.json").write_text(
        json.dumps({"arch_type": "cross_encoder"}), encoding="utf-8"
    )

    jina2_dir = tmp_path / "jina-reranker-v2-base-multilingual"
    jina2_dir.mkdir()
    (jina2_dir / "model_manifest.json").write_text(
        json.dumps({"arch_type": "cross_encoder"}), encoding="utf-8"
    )

    monkeypatch.setattr(config, "RERANKER_ROOT_DIR", str(tmp_path))
    try:
        monkeypatch.setenv("RERANKER_MODEL", "jinaai/jina-reranker-v3.5")
        config.refresh_model_config()
        assert (
            config.CONFIDENCE_THRESHOLD_HIGH,
            config.CONFIDENCE_THRESHOLD_MEDIUM,
            config.RERANKER_THRESHOLD,
            config.RERANKER_MARGIN,
        ) == (0.62, 0.35, 0.30, 0.15)

        monkeypatch.setenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
        config.refresh_model_config()
        assert (
            config.CONFIDENCE_THRESHOLD_HIGH,
            config.CONFIDENCE_THRESHOLD_MEDIUM,
            config.RERANKER_THRESHOLD,
            config.RERANKER_MARGIN,
        ) == (0.78, 0.45, 0.50, 0.12)

        monkeypatch.setenv("RERANKER_MODEL", "jinaai/jina-reranker-v2-base-multilingual")
        config.refresh_model_config()
        assert (
            config.CONFIDENCE_THRESHOLD_HIGH,
            config.CONFIDENCE_THRESHOLD_MEDIUM,
            config.RERANKER_THRESHOLD,
            config.RERANKER_MARGIN,
        ) == (0.65, 0.35, 0.35, 0.12)
    finally:
        monkeypatch.undo()
        config.refresh_model_config()


def test_resolve_onnx_artifact_and_providers_auto_gpu_and_cpu_fallback(tmp_path, monkeypatch):
    from engine.core import config

    fp32 = tmp_path / "model.onnx"
    fp16 = tmp_path / "model_fp16.onnx"
    int8 = tmp_path / "model_int8.onnx"
    int8.write_bytes(b"int8")

    # 1. No CUDA -> always selects INT8 + CPUExecutionProvider in auto mode
    monkeypatch.setattr(config, "check_cuda_provider_available", lambda: False)
    fp16.write_bytes(b"fp16")
    path, provs, prec = config.resolve_onnx_artifact_and_providers(fp32, fp16, int8, "auto")
    assert path == int8
    assert provs == ["CPUExecutionProvider"]
    assert prec == "int8"

    # 2. CUDA available + FP16 exists -> selects FP16 + CUDAExecutionProvider in auto mode
    monkeypatch.setattr(config, "check_cuda_provider_available", lambda: True)
    path_gpu, provs_gpu, prec_gpu = config.resolve_onnx_artifact_and_providers(
        fp32, fp16, int8, "auto"
    )
    assert path_gpu == fp16
    assert provs_gpu == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert prec_gpu == "fp16"

    # 3. CUDA available but only INT8 exists on disk -> falls back cleanly to INT8 + CPUExecutionProvider
    fp16.unlink()
    path_fb, provs_fb, prec_fb = config.resolve_onnx_artifact_and_providers(
        fp32, fp16, int8, "auto"
    )
    assert path_fb == int8
    assert provs_fb == ["CPUExecutionProvider"]
    assert prec_fb == "int8"

    # 4. Explicit int8 mode forces CPU INT8 even if CUDA and FP16 exist
    fp16.write_bytes(b"fp16")
    path_i8, provs_i8, prec_i8 = config.resolve_onnx_artifact_and_providers(
        fp32, fp16, int8, "int8"
    )
    assert path_i8 == int8
    assert provs_i8 == ["CPUExecutionProvider"]
    assert prec_i8 == "int8"


def test_refresh_model_config_resets_when_env_unset(monkeypatch, tmp_path):
    from engine.core import config

    monkeypatch.setattr(config, "EMBEDDING_ROOT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "RERANKER_ROOT_DIR", str(tmp_path))
    monkeypatch.setenv("EMBEDDING_MODEL", "custom/emb-x")
    monkeypatch.setenv("EMBEDDING_POOLING", "cls")
    monkeypatch.setenv("RERANKER_MODEL", "custom/rerank-x")
    try:
        config.refresh_model_config()
        assert config.EMBEDDING_MODEL == "custom/emb-x"
        assert config.EMBEDDING_POOLING == "cls"
        assert config.RERANKER_MODEL == "custom/rerank-x"

        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
        monkeypatch.delenv("EMBEDDING_POOLING", raising=False)
        monkeypatch.delenv("RERANKER_MODEL", raising=False)
        config.refresh_model_config()
        assert config.EMBEDDING_MODEL
        assert config.EMBEDDING_POOLING
        assert config.RERANKER_MODEL
    finally:
        monkeypatch.undo()
        config.refresh_model_config()


def test_embedding_engine_empty_and_left_padded_pooling():
    import threading
    from collections import OrderedDict
    from unittest.mock import MagicMock

    eng = EmbeddingModel.__new__(EmbeddingModel)
    eng.has_learned_sparse = False
    eng.pooling = "last_token"
    eng._cache_lock = threading.RLock()
    eng._str_cache = OrderedDict()

    seen_batches = []

    def _tok(texts, **_kw):
        seen_batches.append(list(texts))
        n = len(texts)
        return {
            "input_ids": np.ones((n, 3), dtype=np.int64),
            "attention_mask": np.array([[1, 1, 0]] * n, dtype=np.int64),
        }

    eng.tokenizer = _tok
    fake_sess = MagicMock()
    fake_sess.get_inputs.return_value = [
        SimpleNamespace(name="input_ids"),
        SimpleNamespace(name="attention_mask"),
    ]
    fake_sess.run.side_effect = lambda _out, inp: [
        np.ones((inp["input_ids"].shape[0], 3, 4), dtype=np.float32) * 2.0
    ]
    eng.bi_session = fake_sess

    res = eng.encode(["", "   ", "hello world"])
    assert seen_batches == [["hello world"]]
    assert np.allclose(res["dense"][0], 0.0)
    assert np.allclose(res["dense"][1], 0.0)
    assert np.isclose(np.linalg.norm(res["dense"][2]), 1.0, atol=1e-4)
    assert "" not in eng._str_cache
    assert "   " not in eng._str_cache
    assert "hello world" in eng._str_cache

    hidden = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    # Row 0 left-padded [0, 1, 1] -> token index 2; Row 1 right-padded [1, 1, 0] -> token index 1
    attn = np.array([[0, 1, 1], [1, 1, 0]], dtype=np.int64)
    pooled = EmbeddingModel.pool_hidden_states(hidden, attn, "last_token")
    assert np.allclose(pooled[0], hidden[0, 2])
    assert np.allclose(pooled[1], hidden[1, 1])


def test_reranker_wrapper_listwise_lbnl():
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")
    from scripts.sync import RerankerONNXWrapper

    class _FakeBackbone(nn.Module):
        def forward(self, input_ids, attention_mask, return_dict=True):
            b, s = input_ids.shape
            return SimpleNamespace(last_hidden_state=torch.ones(b, s, 4, dtype=torch.float32))

    class _FakeLBNLModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _FakeBackbone()
            self.projector = nn.Linear(4, 2, bias=False)
            nn.init.constant_(self.projector.weight, 0.5)

    wrapper = RerankerONNXWrapper(_FakeLBNLModel(), arch_type="listwise_lbnl")
    wrapper.eval()
    out = wrapper(torch.ones((1, 3), dtype=torch.long), torch.ones((1, 3), dtype=torch.long))
    assert out.shape == (1, 3, 2)
    assert torch.allclose(out, torch.full((1, 3, 2), 2.0))


def test_sync_models_single_target_subprocess_and_cache_wipe(monkeypatch, tmp_path):
    from unittest.mock import MagicMock

    import scripts.sync as sync_mod
    from engine.core import config

    monkeypatch.setattr(config, "RERANKER_ONNX_INT8", str(tmp_path / "missing_rerank_int8.onnx"))
    monkeypatch.setattr(sync_mod, "export_reranker", MagicMock(return_value=True))
    monkeypatch.setattr(sync_mod, "export_embedding", MagicMock(return_value=False))
    monkeypatch.setattr(sync_mod, "export_gliner", MagicMock())
    monkeypatch.setattr(sync_mod, "wipe_embedding_caches", MagicMock())
    monkeypatch.setattr(config, "refresh_model_config", MagicMock())
    mock_sub = MagicMock()
    monkeypatch.setattr(sync_mod.subprocess, "run", mock_sub)

    sync_mod.sync_models(mode="rebuild", target="reranker")

    mock_sub.assert_called_once()
    cmd = mock_sub.call_args.args[0]
    assert "--models" in cmd
    assert cmd[cmd.index("--models") + 1] == "quantize"
    assert cmd[cmd.index("--model-target") + 1] == "reranker"
    sync_mod.wipe_embedding_caches.assert_not_called()
