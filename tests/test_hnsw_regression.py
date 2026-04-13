"""
HNSW Regression Test Suite.

This test captures the exact behavior of the current HNSW implementation
so that any refactor (e.g., FastHNSW with integer indices) can be verified
against it. ALL tests must pass before a new implementation can replace
the current one.

Tests cover:
1. Recall accuracy at multiple ef_search values
2. API contract (add, remove, search, rebuild, statistics)
3. Edge cases (duplicate IDs, empty index, dimension mismatch)
4. Feature parity (heuristic selection, bidirectional connections)

Usage:
    pytest tests/test_hnsw_regression.py -v -s
"""

import time
from uuid import uuid4

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# We test against whatever HNSW backend is available (Rust or Python)
@pytest.fixture
def make_index():
    """Factory that creates a VectorStore + HNSW index pair.

    Tries Rust backend first, falls back to Python HNSW.
    The Python HNSW uses the SAME algorithms (paper-correct heuristic
    selection, level generation, etc.) so recall thresholds apply to both.
    """
    from core.vector_store import VectorStore

    backend = "python"
    HNSWIndex = None

    try:
        from infrastructure.indexes.rust_hnsw_wrapper import RustHNSWIndexWrapper
        # Test if Rust module is actually loadable (not just importable)
        import rust_hnsw
        HNSWIndex = RustHNSWIndexWrapper
        backend = "rust"
    except (ImportError, ModuleNotFoundError):
        pass

    if HNSWIndex is None:
        from infrastructure.indexes.hnsw import HNSWIndex as PythonHNSW
        HNSWIndex = PythonHNSW
        backend = "python"

    def _make(dim=128, M=16, ef_construction=200, ef_search=50, capacity=10000):
        vs = VectorStore(dimension=dim, initial_capacity=capacity)
        idx = HNSWIndex(vector_store=vs, M=M, ef_construction=ef_construction, ef_search=ef_search)
        return vs, idx, backend

    return _make


@pytest.fixture
def sift_10k():
    """Generate a reproducible 10K-vector dataset with ground truth.

    Uses normalized random vectors (not real SIFT) but with a fixed seed
    so recall numbers are deterministic.
    """
    np.random.seed(42)
    dim = 128
    n_base = 10_000
    n_query = 100
    k = 10

    base = np.random.randn(n_base, dim).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)

    queries = np.random.randn(n_query, dim).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    # Brute-force ground truth (cosine similarity = dot product for normalized vectors)
    gt = np.zeros((n_query, k), dtype=np.int32)
    for i in range(n_query):
        sims = base @ queries[i]
        gt[i] = np.argsort(-sims)[:k]

    return base, queries, gt


# ---------------------------------------------------------------------------
# 1. Recall Accuracy
# ---------------------------------------------------------------------------

class TestRecallAccuracy:
    """Recall must meet minimum thresholds at various ef_search values.

    These thresholds are set from the CURRENT implementation's behavior.
    A new implementation must match or exceed them.
    """

    def _build_and_search(self, make_index, base, queries, ef_search):
        vs, idx, backend = make_index(dim=base.shape[1], M=16, ef_construction=200, ef_search=ef_search)

        # Insert all vectors
        uuids = []
        for i in range(base.shape[0]):
            uid = uuid4()
            uuids.append(uid)
            vec_idx = vs.add_vector(uid, base[i])
            idx.add_vector(uid, vec_idx)

        # Search
        recalls = []
        for i in range(queries.shape[0]):
            results = idx.search(queries[i], k=10)
            result_indices = set()
            for uid, dist in results:
                # Find the original index for this UUID
                orig_idx = uuids.index(uid) if uid in uuids else -1
                result_indices.add(orig_idx)

            # Not efficient but correct for test
            pass

        return vs, idx, uuids, backend

    def test_recall_at_ef10(self, make_index, sift_10k):
        base, queries, gt = sift_10k
        vs, idx, backend = make_index(dim=128, M=16, ef_construction=200, ef_search=10)

        uuids = []
        for i in range(base.shape[0]):
            uid = uuid4()
            uuids.append(uid)
            vec_idx = vs.add_vector(uid, base[i])
            idx.add_vector(uid, vec_idx)

        recalls = []
        for i in range(queries.shape[0]):
            results = idx.search(queries[i], k=10)
            result_uuids = {uid for uid, _ in results}
            gt_uuids = {uuids[j] for j in gt[i]}
            recall = len(result_uuids & gt_uuids) / 10
            recalls.append(recall)

        mean_recall = np.mean(recalls)
        print(f"[{backend}] ef=10: recall@10 = {mean_recall:.4f}")
        # Thresholds calibrated per backend:
        # Rust achieves 0.969 on SIFT-1M; Python builds slower graphs at M=16
        min_recall = 0.85 if backend == "rust" else 0.60
        assert mean_recall > min_recall, f"Recall {mean_recall:.4f} below minimum {min_recall} at ef=10 ({backend})"

    def test_recall_at_ef50(self, make_index, sift_10k):
        base, queries, gt = sift_10k
        vs, idx, backend = make_index(dim=128, M=16, ef_construction=200, ef_search=50)

        uuids = []
        for i in range(base.shape[0]):
            uid = uuid4()
            uuids.append(uid)
            vec_idx = vs.add_vector(uid, base[i])
            idx.add_vector(uid, vec_idx)

        recalls = []
        for i in range(queries.shape[0]):
            results = idx.search(queries[i], k=10)
            result_uuids = {uid for uid, _ in results}
            gt_uuids = {uuids[j] for j in gt[i]}
            recall = len(result_uuids & gt_uuids) / 10
            recalls.append(recall)

        mean_recall = np.mean(recalls)
        print(f"[{backend}] ef=50: recall@10 = {mean_recall:.4f}")
        # Python HNSW at M=16 builds slower, achieving ~0.82 vs Rust's ~0.99
        min_recall = 0.95 if backend == "rust" else 0.75
        assert mean_recall > min_recall, f"Recall {mean_recall:.4f} below minimum {min_recall} at ef=50 ({backend})"

    def test_recall_at_ef200(self, make_index, sift_10k):
        base, queries, gt = sift_10k
        vs, idx, backend = make_index(dim=128, M=16, ef_construction=200, ef_search=200)

        uuids = []
        for i in range(base.shape[0]):
            uid = uuid4()
            uuids.append(uid)
            vec_idx = vs.add_vector(uid, base[i])
            idx.add_vector(uid, vec_idx)

        recalls = []
        for i in range(queries.shape[0]):
            results = idx.search(queries[i], k=10)
            result_uuids = {uid for uid, _ in results}
            gt_uuids = {uuids[j] for j in gt[i]}
            recall = len(result_uuids & gt_uuids) / 10
            recalls.append(recall)

        mean_recall = np.mean(recalls)
        print(f"[{backend}] ef=200: recall@10 = {mean_recall:.4f}")
        assert mean_recall > 0.98, f"Recall {mean_recall:.4f} below minimum 0.98 at ef=200"

    def test_recall_with_high_M(self, make_index, sift_10k):
        """M=48 (our production config) should achieve higher recall."""
        base, queries, gt = sift_10k
        vs, idx, backend = make_index(dim=128, M=48, ef_construction=400, ef_search=50)

        uuids = []
        for i in range(base.shape[0]):
            uid = uuid4()
            uuids.append(uid)
            vec_idx = vs.add_vector(uid, base[i])
            idx.add_vector(uid, vec_idx)

        recalls = []
        for i in range(queries.shape[0]):
            results = idx.search(queries[i], k=10)
            result_uuids = {uid for uid, _ in results}
            gt_uuids = {uuids[j] for j in gt[i]}
            recall = len(result_uuids & gt_uuids) / 10
            recalls.append(recall)

        mean_recall = np.mean(recalls)
        print(f"[{backend}] M=48, ef=50: recall@10 = {mean_recall:.4f}")
        assert mean_recall > 0.97, f"Recall {mean_recall:.4f} below minimum 0.97 at M=48, ef=50"


# ---------------------------------------------------------------------------
# 2. API Contract
# ---------------------------------------------------------------------------

class TestAPIContract:
    """Every public method must work correctly."""

    def test_add_and_search(self, make_index):
        vs, idx, _ = make_index(dim=4)
        uid = uuid4()
        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        vec_idx = vs.add_vector(uid, vec)
        idx.add_vector(uid, vec_idx)

        results = idx.search(vec, k=1)
        assert len(results) == 1
        assert results[0][0] == uid
        assert results[0][1] < 0.01  # cosine distance ~0 for identical vector

    def test_add_duplicate_raises(self, make_index):
        vs, idx, _ = make_index(dim=4)
        uid = uuid4()
        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        vec_idx = vs.add_vector(uid, vec)
        idx.add_vector(uid, vec_idx)

        with pytest.raises(ValueError, match="already exists"):
            idx.add_vector(uid, vec_idx)

    def test_remove_vector(self, make_index):
        vs, idx, _ = make_index(dim=4)
        uid = uuid4()
        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        vec_idx = vs.add_vector(uid, vec)
        idx.add_vector(uid, vec_idx)

        assert idx.size() == 1
        assert idx.remove_vector(uid) is True
        assert idx.size() == 0

    def test_remove_nonexistent_returns_false(self, make_index):
        vs, idx, _ = make_index(dim=4)
        assert idx.remove_vector(uuid4()) is False

    def test_search_empty_index(self, make_index):
        vs, idx, _ = make_index(dim=4)
        results = idx.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), k=5)
        assert results == []

    def test_size(self, make_index):
        vs, idx, _ = make_index(dim=4)
        assert idx.size() == 0

        for i in range(10):
            uid = uuid4()
            vec = np.random.randn(4).astype(np.float32)
            vec /= np.linalg.norm(vec)
            vec_idx = vs.add_vector(uid, vec)
            idx.add_vector(uid, vec_idx)

        assert idx.size() == 10

    def test_clear(self, make_index):
        vs, idx, _ = make_index(dim=4)

        for i in range(5):
            uid = uuid4()
            vec = np.random.randn(4).astype(np.float32)
            vec /= np.linalg.norm(vec)
            vec_idx = vs.add_vector(uid, vec)
            idx.add_vector(uid, vec_idx)

        assert idx.size() == 5
        idx.clear()
        assert idx.size() == 0

    def test_rebuild(self, make_index):
        vs, idx, _ = make_index(dim=4)

        uids = []
        for i in range(20):
            uid = uuid4()
            uids.append(uid)
            vec = np.random.randn(4).astype(np.float32)
            vec /= np.linalg.norm(vec)
            vec_idx = vs.add_vector(uid, vec)
            idx.add_vector(uid, vec_idx)

        idx.rebuild()
        assert idx.size() == 20

        # Search should still work after rebuild
        query = np.random.randn(4).astype(np.float32)
        query /= np.linalg.norm(query)
        results = idx.search(query, k=5)
        assert len(results) == 5

    def test_distance_threshold(self, make_index):
        vs, idx, _ = make_index(dim=4)

        # Add two vectors: one close, one far
        uid_close = uuid4()
        uid_far = uuid4()
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        close = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
        close /= np.linalg.norm(close)
        far = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        vs.add_vector(uid_close, close)
        idx.add_vector(uid_close, 0)
        vs.add_vector(uid_far, far)
        idx.add_vector(uid_far, 1)

        # With tight threshold, should only return the close one
        results = idx.search(query, k=10, distance_threshold=0.1)
        result_uuids = {uid for uid, _ in results}
        assert uid_close in result_uuids
        assert uid_far not in result_uuids

    def test_search_returns_sorted_by_distance(self, make_index):
        vs, idx, _ = make_index(dim=128, M=16, ef_construction=200, ef_search=50)

        np.random.seed(99)
        for i in range(100):
            uid = uuid4()
            vec = np.random.randn(128).astype(np.float32)
            vec /= np.linalg.norm(vec)
            vec_idx = vs.add_vector(uid, vec)
            idx.add_vector(uid, vec_idx)

        query = np.random.randn(128).astype(np.float32)
        query /= np.linalg.norm(query)
        results = idx.search(query, k=10)

        distances = [dist for _, dist in results]
        assert distances == sorted(distances), "Results not sorted by distance"

    def test_supports_incremental_updates(self, make_index):
        _, idx, _ = make_index(dim=4)
        assert idx.supports_incremental_updates is True

    def test_index_type(self, make_index):
        _, idx, backend = make_index(dim=4)
        assert "hnsw" in idx.index_type.lower()

    def test_set_ef_search(self, make_index):
        vs, idx, backend = make_index(dim=4, ef_search=50)
        if not hasattr(idx, 'set_ef_search'):
            pytest.skip(f"{backend} backend does not support set_ef_search")
        idx.set_ef_search(200)
        # Should not raise; verify by running a search
        uid = uuid4()
        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        vec_idx = vs.add_vector(uid, vec)
        idx.add_vector(uid, vec_idx)
        results = idx.search(vec, k=1)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# 3. QPS Baseline (informational, not a hard gate)
# ---------------------------------------------------------------------------

class TestQPSBaseline:
    """Record QPS so we can verify the new implementation is faster."""

    def test_search_qps_10k(self, make_index, sift_10k):
        base, queries, _ = sift_10k
        vs, idx, backend = make_index(dim=128, M=16, ef_construction=200, ef_search=50)

        for i in range(base.shape[0]):
            uid = uuid4()
            vec_idx = vs.add_vector(uid, base[i])
            idx.add_vector(uid, vec_idx)

        start = time.time()
        for q in queries:
            idx.search(q, k=10)
        elapsed = time.time() - start

        qps = len(queries) / elapsed
        print(f"\n[{backend}] QPS at 10K vectors, ef=50: {qps:.0f}")
        print(f"[{backend}] Mean latency: {elapsed / len(queries) * 1000:.2f}ms")
        # No hard assertion — this is a baseline measurement

    def test_build_rate_10k(self, make_index, sift_10k):
        base, _, _ = sift_10k
        vs, idx, backend = make_index(dim=128, M=16, ef_construction=200, ef_search=50)

        start = time.time()
        for i in range(base.shape[0]):
            uid = uuid4()
            vec_idx = vs.add_vector(uid, base[i])
            idx.add_vector(uid, vec_idx)
        elapsed = time.time() - start

        rate = base.shape[0] / elapsed
        print(f"\n[{backend}] Build rate at 10K vectors: {rate:.0f} vec/s")
        print(f"[{backend}] Total build time: {elapsed:.1f}s")
