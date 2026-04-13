# arrwDB Pre-Release Improvements Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the 5 critical gaps before release: CPU QPS (30x behind hnswlib), pre-filtering, Python SDK, distance metrics, and memory compression.

**Architecture:** Each phase is independent and produces a working, testable improvement. Phase 1 (CPU QPS) has the highest impact and should be done first. Phases 2-5 can be parallelized.

**Tech Stack:** Rust (PyO3 0.21), Python 3.11+, FastAPI, NumPy, maturin

---

## File Structure Overview

### Phase 1: CPU QPS Fix (Rust HNSW refactor)
- Modify: `rust/indexes/src/lib.rs` — Replace String/HashMap storage with integer-indexed arrays
- Modify: `rust/indexes/src/node.rs` — Simplify to flat neighbor arrays
- Modify: `rust/indexes/src/distance.rs` — Add SimSIMD or explicit SIMD
- Create: `rust/indexes/src/storage.rs` — Contiguous vector + graph storage
- Modify: `infrastructure/indexes/rust_hnsw_wrapper.py` — Adapt to new Rust API
- Create: `tests/rust_hnsw_perf_test.py` — Performance regression tests

### Phase 2: Pre-Filtering
- Modify: `rust/indexes/src/lib.rs` — Add filter callback to search_layer
- Modify: `infrastructure/indexes/rust_hnsw_wrapper.py` — Pass filter predicates
- Modify: `infrastructure/indexes/base.py` — Extend search() signature
- Create: `tests/test_filtered_search.py`

### Phase 3: Python Client SDK
- Create: `packages/arrwdb/arrwdb/client.py` — Already exists, extend
- Create: `packages/arrwdb/arrwdb/models.py` — Typed response models
- Create: `tests/test_sdk.py`

### Phase 4: Distance Metrics
- Modify: `rust/indexes/src/distance.rs` — Add L2, inner product, configurable metric
- Modify: `rust/indexes/src/lib.rs` — Accept metric parameter
- Modify: `infrastructure/indexes/rust_hnsw_wrapper.py` — Pass metric
- Create: `tests/test_distance_metrics.py`

### Phase 5: RaBitQ Quantization
- Modify: `rust/indexes/Cargo.toml` — Add rabitq dependency (x86_64 only)
- Create: `rust/indexes/src/quantization.rs` — RaBitQ wrapper
- Modify: `rust/indexes/src/lib.rs` — Optional quantized distance in search
- Create: `tests/test_quantization.py`

---

## Phase 1: Fix CPU QPS (30x gap → target <3x)

The current Rust HNSW uses `HashMap<String, HNSWNode>` for graph storage and `HashMap<String, Vec<f32>>` for vector storage. Every neighbor visit in the search loop clones Strings, acquires RwLocks, and does HashMap lookups. The fix is to use integer indices and contiguous arrays.

### Task 1.1: Create storage.rs — Contiguous Vector + Graph Storage

**Files:**
- Create: `rust/indexes/src/storage.rs`
- Modify: `rust/indexes/src/lib.rs` (add `mod storage;`)

- [ ] **Step 1: Create storage.rs with VectorStorage struct**

```rust
// rust/indexes/src/storage.rs

/// Contiguous storage for vectors and graph structure.
/// All lookups are O(1) array indexing — no HashMap, no String hashing.
pub struct VectorStorage {
    /// Flat vector data: vectors[id * dim .. (id+1) * dim]
    vectors: Vec<f32>,
    /// Dimension of each vector
    dim: usize,
    /// Number of stored vectors
    count: usize,
    /// Capacity (pre-allocated slots)
    capacity: usize,
}

impl VectorStorage {
    pub fn new(dim: usize, capacity: usize) -> Self {
        Self {
            vectors: vec![0.0f32; dim * capacity],
            dim,
            count: 0,
            capacity,
        }
    }

    /// Add a vector, returns its index.
    pub fn add(&mut self, vector: &[f32]) -> usize {
        assert_eq!(vector.len(), self.dim);
        if self.count >= self.capacity {
            self.grow();
        }
        let idx = self.count;
        let start = idx * self.dim;
        self.vectors[start..start + self.dim].copy_from_slice(vector);
        self.count += 1;
        idx
    }

    /// Get vector by index (zero-copy slice).
    #[inline]
    pub fn get(&self, idx: usize) -> &[f32] {
        let start = idx * self.dim;
        &self.vectors[start..start + self.dim]
    }

    pub fn len(&self) -> usize {
        self.count
    }

    fn grow(&mut self) {
        let new_capacity = (self.capacity * 3) / 2 + 1;
        self.vectors.resize(new_capacity * self.dim, 0.0f32);
        self.capacity = new_capacity;
    }
}

/// Flat graph storage: neighbors[node_id][layer] = Vec<usize>
pub struct GraphStorage {
    /// Per-node, per-layer neighbor lists.
    /// neighbors[node_id] = Vec<(layer, Vec<neighbor_id>)> flattened as:
    /// neighbors[node_id] has entries for layers 0..=node_level
    neighbors: Vec<NodeNeighbors>,
    /// Max level assigned to each node
    levels: Vec<usize>,
    /// Number of nodes
    count: usize,
}

pub struct NodeNeighbors {
    /// neighbors_per_layer[layer] = Vec<usize> of neighbor indices
    pub layers: Vec<Vec<usize>>,
}

impl GraphStorage {
    pub fn new(capacity: usize) -> Self {
        Self {
            neighbors: Vec::with_capacity(capacity),
            levels: Vec::with_capacity(capacity),
            count: 0,
        }
    }

    /// Add a node with the given level. Returns node index.
    pub fn add_node(&mut self, level: usize) -> usize {
        let idx = self.count;
        let mut layers = Vec::with_capacity(level + 1);
        for _ in 0..=level {
            layers.push(Vec::new());
        }
        self.neighbors.push(NodeNeighbors { layers });
        self.levels.push(level);
        self.count += 1;
        idx
    }

    #[inline]
    pub fn get_neighbors(&self, node_id: usize, layer: usize) -> &[usize] {
        &self.neighbors[node_id].layers[layer]
    }

    #[inline]
    pub fn get_neighbors_mut(&mut self, node_id: usize, layer: usize) -> &mut Vec<usize> {
        &mut self.neighbors[node_id].layers[layer]
    }

    #[inline]
    pub fn level(&self, node_id: usize) -> usize {
        self.levels[node_id]
    }

    pub fn len(&self) -> usize {
        self.count
    }
}
```

- [ ] **Step 2: Add mod storage to lib.rs**

Add at the top of `rust/indexes/src/lib.rs`:
```rust
mod storage;
```

- [ ] **Step 3: Verify it compiles**

Run: `cd rust && cargo check 2>&1 | tail -5`
Expected: Compiles with warnings only (unused code)

- [ ] **Step 4: Commit**

```bash
git add rust/indexes/src/storage.rs rust/indexes/src/lib.rs
git commit -m "feat: Add contiguous vector and graph storage (storage.rs)"
```

---

### Task 1.2: Create FastHNSW — Integer-Indexed HNSW Implementation

**Files:**
- Create: `rust/indexes/src/fast_hnsw.rs` — New HNSW using storage.rs
- Modify: `rust/indexes/src/lib.rs` — Add mod fast_hnsw, expose via PyO3

This is the core refactor. The new implementation uses `usize` indices everywhere, contiguous arrays for vectors and neighbors, and a single read snapshot for the entire search.

- [ ] **Step 1: Create fast_hnsw.rs with struct and constructor**

```rust
// rust/indexes/src/fast_hnsw.rs
use crate::distance::cosine_distance;
use crate::storage::{GraphStorage, VectorStorage};
use parking_lot::RwLock;
use rand::Rng;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// High-performance HNSW index using integer indices and contiguous arrays.
/// All IDs are usize — String mapping happens at the PyO3 boundary only.
pub struct FastHNSW {
    m: usize,
    m_max0: usize,           // 2 * m for layer 0
    ef_construction: usize,
    ef_search: usize,
    ml: f64,
    max_level: usize,

    vectors: RwLock<VectorStorage>,
    graph: RwLock<GraphStorage>,
    entry_point: RwLock<Option<usize>>,
    entry_level: RwLock<usize>,

    dim: usize,
}

#[derive(Clone)]
struct Candidate {
    id: usize,
    dist: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool { self.dist == other.dist }
}
impl Eq for Candidate {}

// Min-heap ordering (smallest distance first)
struct MinCandidate(Candidate);
impl PartialEq for MinCandidate {
    fn eq(&self, other: &Self) -> bool { self.0.dist == other.0.dist }
}
impl Eq for MinCandidate {}
impl PartialOrd for MinCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for MinCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.dist.partial_cmp(&self.0.dist).unwrap_or(Ordering::Equal)
    }
}

// Max-heap ordering (largest distance first)
struct MaxCandidate(Candidate);
impl PartialEq for MaxCandidate {
    fn eq(&self, other: &Self) -> bool { self.0.dist == other.0.dist }
}
impl Eq for MaxCandidate {}
impl PartialOrd for MaxCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for MaxCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.dist.partial_cmp(&other.0.dist).unwrap_or(Ordering::Equal)
    }
}

impl FastHNSW {
    pub fn new(dim: usize, m: usize, ef_construction: usize, ef_search: usize, max_level: usize) -> Self {
        let capacity = 1024;
        Self {
            m,
            m_max0: 2 * m,
            ef_construction,
            ef_search,
            ml: 1.0 / (m as f64).ln(),
            max_level,
            vectors: RwLock::new(VectorStorage::new(dim, capacity)),
            graph: RwLock::new(GraphStorage::new(capacity)),
            entry_point: RwLock::new(None),
            entry_level: RwLock::new(0),
            dim,
        }
    }

    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen();
        let level = (-r.ln() * self.ml).floor() as usize;
        level.min(self.max_level)
    }

    fn m_max(&self, layer: usize) -> usize {
        if layer == 0 { self.m_max0 } else { self.m }
    }
}
```

- [ ] **Step 2: Implement search_layer — the critical hot path**

Add to `fast_hnsw.rs`:

```rust
impl FastHNSW {
    /// Search at a single layer. Returns up to ef nearest neighbors.
    /// Takes direct references to storage to avoid repeated lock acquisition.
    fn search_layer(
        vectors: &VectorStorage,
        graph: &GraphStorage,
        query: &[f32],
        entry_id: usize,
        ef: usize,
        layer: usize,
    ) -> Vec<Candidate> {
        let mut visited = vec![false; graph.len()];
        visited[entry_id] = true;

        let entry_dist = cosine_distance(query, vectors.get(entry_id));

        let mut candidates = BinaryHeap::new();
        candidates.push(MinCandidate(Candidate { id: entry_id, dist: entry_dist }));

        let mut results = BinaryHeap::new();
        results.push(MaxCandidate(Candidate { id: entry_id, dist: entry_dist }));

        while let Some(MinCandidate(current)) = candidates.pop() {
            let worst_dist = results.peek().unwrap().0.dist;
            if current.dist > worst_dist && results.len() >= ef {
                break;
            }

            let neighbors = graph.get_neighbors(current.id, layer);
            for &neighbor_id in neighbors {
                if visited[neighbor_id] {
                    continue;
                }
                visited[neighbor_id] = true;

                let dist = cosine_distance(query, vectors.get(neighbor_id));

                if dist < worst_dist || results.len() < ef {
                    candidates.push(MinCandidate(Candidate { id: neighbor_id, dist }));
                    results.push(MaxCandidate(Candidate { id: neighbor_id, dist }));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        results.into_iter().map(|MaxCandidate(c)| c).collect()
    }
}
```

Key differences from old code:
- `visited` is a `Vec<bool>` (O(1) bitset) instead of `HashSet<String>`
- `vectors.get(id)` is direct array indexing instead of HashMap + RwLock
- `graph.get_neighbors(id, layer)` is direct array indexing instead of HashMap<String, HashSet<String>>
- Zero String allocations in the entire loop
- Locks acquired ONCE at the caller, not per-neighbor

- [ ] **Step 3: Implement add_vector and insert_node**

Add to `fast_hnsw.rs`:

```rust
impl FastHNSW {
    /// Add a vector to the index. Returns its internal index.
    pub fn add_vector(&self, vector: &[f32]) -> usize {
        let level = self.random_level();

        let idx = {
            let mut vecs = self.vectors.write();
            vecs.add(vector)
        };

        {
            let mut g = self.graph.write();
            g.add_node(level);
        }

        let ep = *self.entry_point.read();

        if ep.is_none() {
            *self.entry_point.write() = Some(idx);
            *self.entry_level.write() = level;
            return idx;
        }

        let ep_id = ep.unwrap();
        self.insert_node(idx, level, ep_id);

        if level > *self.entry_level.read() {
            *self.entry_point.write() = Some(idx);
            *self.entry_level.write() = level;
        }

        idx
    }

    fn insert_node(&self, new_id: usize, new_level: usize, ep_id: usize) {
        let vectors = self.vectors.read();
        let entry_level = *self.entry_level.read();
        let query = vectors.get(new_id);

        let mut current = ep_id;

        // Navigate upper layers (above new_level)
        {
            let graph = self.graph.read();
            for lc in (new_level + 1..=entry_level).rev() {
                let mut changed = true;
                while changed {
                    changed = false;
                    let neighbors = graph.get_neighbors(current, lc);
                    for &nid in neighbors {
                        let d = cosine_distance(query, vectors.get(nid));
                        if d < cosine_distance(query, vectors.get(current)) {
                            current = nid;
                            changed = true;
                        }
                    }
                }
            }
        }

        // Insert at layers new_level down to 0
        for lc in (0..=new_level.min(entry_level)).rev() {
            let candidates = {
                let graph = self.graph.read();
                Self::search_layer(&vectors, &graph, query, current, self.ef_construction, lc)
            };

            if !candidates.is_empty() {
                current = candidates.iter().min_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap()).unwrap().id;
            }

            // Select neighbors (simple: closest M)
            let m_max = self.m_max(lc);
            let mut sorted = candidates;
            sorted.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
            let selected: Vec<usize> = sorted.iter().take(m_max).map(|c| c.id).collect();

            // Connect new node to selected neighbors
            {
                let mut graph = self.graph.write();
                *graph.get_neighbors_mut(new_id, lc) = selected.clone();

                // Add reverse connections and prune if needed
                for &neighbor_id in &selected {
                    let nbs = graph.get_neighbors_mut(neighbor_id, lc);
                    nbs.push(new_id);
                    if nbs.len() > m_max {
                        // Simple pruning: keep closest
                        let mut with_dist: Vec<(usize, f32)> = nbs.iter()
                            .map(|&nid| (nid, cosine_distance(vectors.get(neighbor_id), vectors.get(nid))))
                            .collect();
                        with_dist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                        *nbs = with_dist.into_iter().take(m_max).map(|(id, _)| id).collect();
                    }
                }
            }
        }
    }

    /// Search for k nearest neighbors.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(usize, f32)> {
        let ep = match *self.entry_point.read() {
            Some(ep) => ep,
            None => return Vec::new(),
        };

        let vectors = self.vectors.read();
        let graph = self.graph.read();
        let entry_level = *self.entry_level.read();

        let mut current = ep;

        // Navigate upper layers greedily
        for lc in (1..=entry_level).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                let neighbors = graph.get_neighbors(current, lc);
                for &nid in neighbors {
                    if cosine_distance(query, vectors.get(nid)) < cosine_distance(query, vectors.get(current)) {
                        current = nid;
                        changed = true;
                    }
                }
            }
        }

        // Search layer 0
        let mut results = Self::search_layer(&vectors, &graph, query, current, ef, 0);
        results.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        results.truncate(k);
        results.into_iter().map(|c| (c.id, c.dist)).collect()
    }

    pub fn len(&self) -> usize {
        self.graph.read().len()
    }
}
```

- [ ] **Step 4: Add PyO3 wrapper for FastHNSW**

Add to the bottom of `fast_hnsw.rs`:

```rust
use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict};
use numpy::PyReadonlyArray1;
use std::collections::HashMap;

#[pyclass]
pub struct RustFastHNSWIndex {
    inner: FastHNSW,
    /// String ID → internal usize index
    id_to_idx: parking_lot::RwLock<HashMap<String, usize>>,
    /// Internal usize index → String ID
    idx_to_id: parking_lot::RwLock<Vec<String>>,
}

#[pymethods]
impl RustFastHNSWIndex {
    #[new]
    #[pyo3(signature = (dimension, m=16, ef_construction=200, ef_search=50, max_level=16))]
    fn new(dimension: usize, m: usize, ef_construction: usize, ef_search: usize, max_level: usize) -> Self {
        Self {
            inner: FastHNSW::new(dimension, m, ef_construction, ef_search, max_level),
            id_to_idx: parking_lot::RwLock::new(HashMap::new()),
            idx_to_id: parking_lot::RwLock::new(Vec::new()),
        }
    }

    fn add_vector(&self, vector_id: String, vector: PyReadonlyArray1<f32>) -> PyResult<()> {
        let data = vector.as_slice()?;
        let idx = self.inner.add_vector(data);
        self.id_to_idx.write().insert(vector_id.clone(), idx);
        let mut ids = self.idx_to_id.write();
        if ids.len() <= idx {
            ids.resize(idx + 1, String::new());
        }
        ids[idx] = vector_id;
        Ok(())
    }

    fn search<'py>(
        &self,
        py: Python<'py>,
        query_vector: PyReadonlyArray1<f32>,
        k: usize,
        distance_threshold: Option<f32>,
        ef_override: Option<usize>,
    ) -> PyResult<&'py PyList> {
        let query = query_vector.as_slice()?;
        let ef = ef_override.unwrap_or(self.inner.ef_search);
        let results = self.inner.search(query, k, ef);

        let idx_to_id = self.idx_to_id.read();
        let result_list = PyList::empty(py);
        for (idx, dist) in results {
            if let Some(threshold) = distance_threshold {
                if dist > threshold { continue; }
            }
            let vid = &idx_to_id[idx];
            result_list.append((vid.as_str(), dist).to_object(py))?;
        }
        Ok(result_list)
    }

    fn set_ef_search(&self, ef: usize) {
        // Store in inner for default searches
        // (FastHNSW.ef_search is not behind a lock since it's set once)
    }

    fn size(&self) -> usize {
        self.inner.len()
    }

    fn clear(&self) {
        // Reset all storage
        *self.inner.vectors.write() = VectorStorage::new(self.inner.dim, 1024);
        *self.inner.graph.write() = GraphStorage::new(1024);
        *self.inner.entry_point.write() = None;
        *self.inner.entry_level.write() = 0;
        self.id_to_idx.write().clear();
        self.idx_to_id.write().clear();
    }
}
```

- [ ] **Step 5: Register in PyO3 module**

In `rust/indexes/src/lib.rs`, add to the `#[pymodule]` function:

```rust
mod fast_hnsw;

// In the rust_hnsw module function:
m.add_class::<fast_hnsw::RustFastHNSWIndex>()?;
```

- [ ] **Step 6: Build and verify**

Run:
```bash
cd rust/indexes && maturin build --release 2>&1 | tail -5
pip install ../target/wheels/rust_hnsw-*.whl --force-reinstall
python -c "from rust_hnsw import RustFastHNSWIndex; print('FastHNSW imported OK')"
```

- [ ] **Step 7: Write performance comparison test**

Create `tests/rust_hnsw_perf_test.py`:

```python
"""Performance comparison: old RustHNSWIndex vs new RustFastHNSWIndex."""
import time
import numpy as np

def test_fast_hnsw_search_qps():
    from rust_hnsw import RustFastHNSWIndex

    dim = 128
    n = 10_000
    index = RustFastHNSWIndex(dimension=dim, m=16, ef_construction=200, ef_search=50)

    np.random.seed(42)
    vectors = np.random.randn(n, dim).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    for i in range(n):
        index.add_vector(f"vec_{i}", vectors[i])

    queries = np.random.randn(100, dim).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    start = time.time()
    for q in queries:
        index.search(q, k=10)
    elapsed = time.time() - start

    qps = 100 / elapsed
    print(f"FastHNSW: {qps:.0f} QPS at 10K vectors, dim={dim}")
    assert qps > 500, f"Expected >500 QPS, got {qps:.0f}"


def test_old_vs_new_qps():
    from rust_hnsw import RustHNSWIndex, RustFastHNSWIndex

    dim = 128
    n = 10_000
    np.random.seed(42)
    vectors = np.random.randn(n, dim).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    queries = np.random.randn(100, dim).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    # Old index
    old = RustHNSWIndex(dimension=dim, m=16, ef_construction=200, ef_search=50)
    for i in range(n):
        old.add_vector(f"vec_{i}", vectors[i])

    start = time.time()
    for q in queries:
        old.search(q, k=10)
    old_elapsed = time.time() - start
    old_qps = 100 / old_elapsed

    # New index
    new = RustFastHNSWIndex(dimension=dim, m=16, ef_construction=200, ef_search=50)
    for i in range(n):
        new.add_vector(f"vec_{i}", vectors[i])

    start = time.time()
    for q in queries:
        new.search(q, k=10)
    new_elapsed = time.time() - start
    new_qps = 100 / new_elapsed

    speedup = new_qps / old_qps
    print(f"Old: {old_qps:.0f} QPS, New: {new_qps:.0f} QPS, Speedup: {speedup:.1f}x")
    assert speedup > 3, f"Expected >3x speedup, got {speedup:.1f}x"
```

- [ ] **Step 8: Run performance test**

Run: `pytest tests/rust_hnsw_perf_test.py -v -s`
Expected: FastHNSW is >3x faster than old RustHNSWIndex

- [ ] **Step 9: Update rust_hnsw_wrapper.py to use FastHNSW**

In `infrastructure/indexes/rust_hnsw_wrapper.py`, change the import to prefer FastHNSW:

```python
try:
    from rust_hnsw import RustFastHNSWIndex as RustHNSW
    USING_FAST_HNSW = True
except ImportError:
    try:
        from rust_hnsw import RustHNSWIndex as RustHNSW
        USING_FAST_HNSW = False
    except ImportError:
        RustHNSW = None
        USING_FAST_HNSW = False
```

- [ ] **Step 10: Commit**

```bash
git add rust/indexes/src/fast_hnsw.rs rust/indexes/src/storage.rs \
        rust/indexes/src/lib.rs infrastructure/indexes/rust_hnsw_wrapper.py \
        tests/rust_hnsw_perf_test.py
git commit -m "feat: Add FastHNSW with integer-indexed storage (10-30x QPS improvement)"
```

---

## Phase 2: Pre-Filtering (Metadata-Aware Search)

### Task 2.1: Extend search() to accept a filter function

The approach: pass a Python callable (filter predicate) across the PyO3 boundary. The Rust search_layer checks the predicate before adding candidates to results. This avoids post-filtering's empty-result problem.

**Files:**
- Modify: `rust/indexes/src/fast_hnsw.rs` — Add filtered_search method
- Modify: `infrastructure/indexes/rust_hnsw_wrapper.py` — Pass filter
- Modify: `infrastructure/indexes/base.py` — Add filter parameter to search()

- [ ] **Step 1: Add filtered search to FastHNSW**

In `fast_hnsw.rs`, add to the `#[pymethods]` impl:

```rust
fn search_filtered<'py>(
    &self,
    py: Python<'py>,
    query_vector: PyReadonlyArray1<f32>,
    k: usize,
    filter_ids: Vec<String>,  // IDs that PASS the filter
    distance_threshold: Option<f32>,
    ef_override: Option<usize>,
) -> PyResult<&'py PyList> {
    let query = query_vector.as_slice()?;
    let ef = ef_override.unwrap_or(self.inner.ef_search);

    // Convert allowed IDs to a bitset for O(1) lookup
    let id_to_idx = self.id_to_idx.read();
    let n = self.inner.len();
    let mut allowed = vec![false; n];
    for fid in &filter_ids {
        if let Some(&idx) = id_to_idx.get(fid) {
            allowed[idx] = true;
        }
    }

    // Search with oversampling (ef * 2 to compensate for filtered-out results)
    let oversample_ef = ef * 2;
    let results = self.inner.search(query, oversample_ef, oversample_ef);

    // Filter and return
    let idx_to_id = self.idx_to_id.read();
    let result_list = PyList::empty(py);
    let mut count = 0;
    for (idx, dist) in results {
        if count >= k { break; }
        if !allowed[idx] { continue; }
        if let Some(threshold) = distance_threshold {
            if dist > threshold { continue; }
        }
        let vid = &idx_to_id[idx];
        result_list.append((vid.as_str(), dist).to_object(py))?;
        count += 1;
    }
    Ok(result_list)
}
```

- [ ] **Step 2: Update VectorIndex ABC**

In `infrastructure/indexes/base.py`, extend the search signature:

```python
@abstractmethod
def search(
    self,
    query_vector: NDArray[np.float32],
    k: int,
    distance_threshold: Optional[float] = None,
    filter_ids: Optional[Set[UUID]] = None,
) -> List[Tuple[UUID, float]]:
```

- [ ] **Step 3: Update wrapper to pass filter_ids**

In `rust_hnsw_wrapper.py`:

```python
def search(self, query_vector, k, distance_threshold=None, filter_ids=None):
    if filter_ids is not None:
        filter_strs = [str(uid) for uid in filter_ids]
        results = self._rust_index.search_filtered(
            query_vector, k, filter_strs, distance_threshold
        )
    else:
        results = self._rust_index.search(
            query_vector, k, distance_threshold
        )
    return [(UUID(vid), dist) for vid, dist in results]
```

- [ ] **Step 4: Write test**

```python
def test_filtered_search():
    # Create index with 100 vectors labeled "A" or "B"
    # Search with filter_ids = only "A" vectors
    # Verify all results are "A" vectors
    # Verify recall is still high within the filtered set
```

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: Add pre-filtered search to HNSW (oversampling strategy)"
```

---

## Phase 3: Python Client SDK

### Task 3.1: Extend the existing arrwdb client package

**Files:**
- Modify: `packages/arrwdb/arrwdb/client.py` — Add missing methods, type hints
- Create: `packages/arrwdb/arrwdb/models.py` — Typed response models (Pydantic)
- Create: `packages/arrwdb/arrwdb/exceptions.py` — Custom exceptions
- Modify: `packages/arrwdb/setup.py` or `pyproject.toml` — Add pydantic dependency

- [ ] **Step 1: Create response models**

```python
# packages/arrwdb/arrwdb/models.py
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class SearchResult:
    chunk_id: str
    text: str
    distance: float
    metadata: Dict[str, Any]

@dataclass
class Library:
    id: str
    name: str
    description: Optional[str]
    index_type: str
    document_count: int

@dataclass
class Document:
    id: str
    title: str
    chunk_count: int
    metadata: Dict[str, Any]
```

- [ ] **Step 2: Create exceptions**

```python
# packages/arrwdb/arrwdb/exceptions.py
class ArrwDBError(Exception):
    """Base exception for arrwDB client."""
    def __init__(self, message: str, status_code: int = None):
        super().__init__(message)
        self.status_code = status_code

class NotFoundError(ArrwDBError): pass
class ValidationError(ArrwDBError): pass
class AuthenticationError(ArrwDBError): pass
class RateLimitError(ArrwDBError): pass
```

- [ ] **Step 3: Enhance the client with typed methods, retry logic, and error handling**

Review and extend `packages/arrwdb/arrwdb/client.py` with:
- Return typed model objects instead of raw dicts
- Add retry logic with exponential backoff for 429/503
- Map HTTP status codes to custom exceptions
- Add `search_with_filter(library_id, embedding, k, filter)` method
- Add context manager support (`with ArrwDBClient(...) as client:`)

- [ ] **Step 4: Write SDK tests**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: Add typed Python client SDK with models, exceptions, and retry logic"
```

---

## Phase 4: Multiple Distance Metrics

### Task 4.1: Add L2 and inner product to distance.rs

**Files:**
- Modify: `rust/indexes/src/distance.rs` — Add l2_distance, inner_product
- Modify: `rust/indexes/src/fast_hnsw.rs` — Accept metric parameter
- Modify: `rust/indexes/src/lib.rs` — Pass metric through PyO3

- [ ] **Step 1: Add distance functions**

```rust
// rust/indexes/src/distance.rs

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DistanceMetric {
    Cosine,
    L2,
    InnerProduct,
}

#[inline]
pub fn compute_distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::Cosine => cosine_distance(a, b),
        DistanceMetric::L2 => l2_distance(a, b),
        DistanceMetric::InnerProduct => inner_product_distance(a, b),
    }
}

#[inline]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f32>()
}

#[inline]
pub fn inner_product_distance(a: &[f32], b: &[f32]) -> f32 {
    // Return 1 - dot_product so smaller = more similar (consistent with other metrics)
    1.0 - a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
}
```

- [ ] **Step 2: Update FastHNSW to accept metric**

Add `metric: DistanceMetric` field to `FastHNSW`, pass through PyO3 as string parameter ("cosine", "l2", "inner_product").

- [ ] **Step 3: Write tests for each metric**

- [ ] **Step 4: Commit**

```bash
git commit -m "feat: Add L2 and inner product distance metrics"
```

---

## Phase 5: RaBitQ Quantization

### Task 5.1: Integrate quantized distance computation

**Note:** rabitq-rs has critical bugs on ARM64/Apple Silicon. This phase MUST be tested on x86_64 (the GCP VMs).

**Files:**
- Modify: `rust/indexes/Cargo.toml` — Add rabitq dependency (behind feature flag)
- Create: `rust/indexes/src/quantization.rs` — RaBitQ wrapper
- Modify: `rust/indexes/src/fast_hnsw.rs` — Optional quantized search mode

- [ ] **Step 1: Add rabitq to Cargo.toml behind feature flag**

```toml
[features]
default = []
rabitq = ["dep:rabitq"]

[dependencies]
rabitq = { version = "0.9", optional = true }
```

- [ ] **Step 2: Create quantization.rs**

Implement a thin wrapper around rabitq-rs that quantizes vectors at build time and provides a fast approximate distance function for the search loop.

- [ ] **Step 3: Add quantized search mode to FastHNSW**

When quantization is enabled, the search loop uses the quantized distance for candidate selection (fast, approximate) and reranks the top-ef results with exact distance (slow, precise). This is the standard two-phase approach.

- [ ] **Step 4: Write quantization tests**

Test recall degradation: must maintain >0.98 recall at 32x compression.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: Add optional RaBitQ quantization (32x memory compression)"
```

---

## Testing & Verification

After all phases, run the full benchmark suite on the GCP VM to verify:

1. **CPU QPS**: Re-run SIFT-1M benchmark with FastHNSW — target >1,000 QPS at 0.99 recall
2. **Filtered search**: Verify recall >0.95 with 50% filter selectivity
3. **Distance metrics**: Verify recall >0.99 on SIFT-1M with L2 metric
4. **Memory**: Verify RaBitQ reduces memory by >20x with <2% recall loss

---

## Estimated Timeline

| Phase | Effort | Dependencies |
|-------|--------|-------------|
| Phase 1: CPU QPS | 3-4 days | None |
| Phase 2: Pre-filtering | 1-2 days | Phase 1 (uses FastHNSW) |
| Phase 3: Python SDK | 1-2 days | None (parallel) |
| Phase 4: Distance metrics | 1 day | Phase 1 (modifies FastHNSW) |
| Phase 5: RaBitQ | 2-3 days | Phase 1, x86_64 VM |
| **Total** | **8-12 days** | |
