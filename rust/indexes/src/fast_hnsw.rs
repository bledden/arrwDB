/// High-performance HNSW using integer indices and contiguous arrays.
///
/// This is a faithful port of the algorithms in lib.rs (RustHNSWIndex),
/// with String/HashMap replaced by usize/Vec for O(1) lookups.
///
/// Preserved from the original:
/// - Paper-correct level generation: floor(-ln(uniform) * ml) where ml = 1/ln(M)
/// - Algorithm 4 diversity-aware neighbor selection with backfill
/// - Bidirectional pruning
/// - remove_vector with neighbor cleanup
/// - rebuild from scratch
/// - batch_search with rayon

use crate::distance::{compute_distance, DistanceMetric};
use crate::storage::{GraphStorage, VectorStorage};
use parking_lot::RwLock;
use rand::Rng;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

// -------------------------------------------------------------------------
// Candidate types for BinaryHeap
// -------------------------------------------------------------------------

#[derive(Clone)]
struct Candidate {
    id: usize,
    dist: f32,
}

/// Min-heap wrapper (smallest distance = highest priority).
struct MinCand(Candidate);

impl PartialEq for MinCand {
    fn eq(&self, other: &Self) -> bool { self.0.dist == other.0.dist }
}
impl Eq for MinCand {}
impl PartialOrd for MinCand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for MinCand {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.dist.partial_cmp(&self.0.dist).unwrap_or(Ordering::Equal)
    }
}

/// Max-heap wrapper (largest distance = highest priority).
struct MaxCand(Candidate);

impl PartialEq for MaxCand {
    fn eq(&self, other: &Self) -> bool { self.0.dist == other.0.dist }
}
impl Eq for MaxCand {}
impl PartialOrd for MaxCand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for MaxCand {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.dist.partial_cmp(&other.0.dist).unwrap_or(Ordering::Equal)
    }
}

// -------------------------------------------------------------------------
// FastHNSW core
// -------------------------------------------------------------------------

pub struct FastHNSW {
    m: usize,
    m_max0: usize,
    ef_construction: usize,
    pub ef_search: usize,
    ml: f64,
    max_level: usize,
    dim: usize,
    metric: DistanceMetric,

    pub vectors: RwLock<VectorStorage>,
    pub graph: RwLock<GraphStorage>,
    entry_point: RwLock<Option<usize>>,
    entry_level: RwLock<usize>,

    /// Tracks which indices are "alive" (not removed). True = alive.
    alive: RwLock<Vec<bool>>,
}

impl FastHNSW {
    pub fn new(
        dim: usize,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        max_level: usize,
        metric: DistanceMetric,
    ) -> Self {
        Self {
            m,
            m_max0: 2 * m,
            ef_construction,
            ef_search,
            ml: 1.0 / (m as f64).ln(),
            max_level,
            dim,
            metric,
            vectors: RwLock::new(VectorStorage::new(dim, 1024)),
            graph: RwLock::new(GraphStorage::new(1024)),
            entry_point: RwLock::new(None),
            entry_level: RwLock::new(0),
            alive: RwLock::new(Vec::new()),
        }
    }

    /// Paper formula: floor(-ln(uniform) * ml) where ml = 1/ln(M)
    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen();
        let level = (-r.ln() * self.ml).floor() as usize;
        level.min(self.max_level)
    }

    fn m_max(&self, layer: usize) -> usize {
        if layer == 0 { self.m_max0 } else { self.m }
    }

    // ---------------------------------------------------------------------
    // search_layer — the critical hot path
    // ---------------------------------------------------------------------

    /// Search at a single layer. Takes direct references (no lock acquisition).
    fn search_layer(
        vectors: &VectorStorage,
        graph: &GraphStorage,
        alive: &[bool],
        query: &[f32],
        entry_id: usize,
        ef: usize,
        layer: usize,
        metric: DistanceMetric,
    ) -> Vec<Candidate> {
        let n = graph.len();
        let mut visited = vec![false; n];
        visited[entry_id] = true;

        let entry_dist = compute_distance(query, vectors.get(entry_id), metric);

        let mut candidates = BinaryHeap::new();
        candidates.push(MinCand(Candidate { id: entry_id, dist: entry_dist }));

        let mut results = BinaryHeap::new();
        results.push(MaxCand(Candidate { id: entry_id, dist: entry_dist }));

        while let Some(MinCand(current)) = candidates.pop() {
            let worst_dist = results.peek().unwrap().0.dist;

            // Explore neighbors FIRST (before termination check) —
            // matches the original behavior that ensures border candidates
            // have their neighbors checked.
            let neighbors = graph.get_neighbors(current.id, layer);
            for &nid in neighbors {
                if visited[nid] || !alive[nid] {
                    continue;
                }
                visited[nid] = true;

                let dist = compute_distance(query, vectors.get(nid), metric);

                if dist < worst_dist || results.len() < ef {
                    candidates.push(MinCand(Candidate { id: nid, dist }));
                    results.push(MaxCand(Candidate { id: nid, dist }));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }

            // Check termination AFTER exploring neighbors
            if let Some(next) = candidates.peek() {
                if next.0.dist > results.peek().unwrap().0.dist && results.len() >= ef {
                    break;
                }
            }
        }

        results.into_iter().map(|MaxCand(c)| c).collect()
    }

    // ---------------------------------------------------------------------
    // select_neighbors_heuristic — Algorithm 4 with diversity + backfill
    // Faithful port of the original select_neighbors_heuristic()
    // ---------------------------------------------------------------------

    fn select_neighbors_heuristic(
        vectors: &VectorStorage,
        mut candidates: Vec<Candidate>,
        m: usize,
        metric: DistanceMetric,
    ) -> Vec<Candidate> {
        if candidates.len() <= m {
            return candidates;
        }

        // Sort by distance ascending (closest first)
        candidates.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));

        let mut selected: Vec<Candidate> = Vec::with_capacity(m);
        let mut discarded: Vec<Candidate> = Vec::new();

        for cand in candidates {
            if selected.len() >= m {
                break;
            }

            // Check diversity: candidate must be closer to query than to
            // any already-selected neighbor
            let cand_vec = vectors.get(cand.id);
            let is_diverse = selected.iter().all(|sel| {
                let sel_vec = vectors.get(sel.id);
                let inter_dist = compute_distance(cand_vec, sel_vec, metric);
                cand.dist <= inter_dist
            });

            if is_diverse {
                selected.push(cand);
            } else {
                discarded.push(cand);
            }
        }

        // Backfill from discarded to guarantee m neighbors
        for cand in discarded {
            if selected.len() >= m {
                break;
            }
            selected.push(cand);
        }

        selected
    }

    // ---------------------------------------------------------------------
    // prune_connections — with bidirectional removal
    // ---------------------------------------------------------------------

    fn prune_connections(
        vectors: &VectorStorage,
        graph: &mut GraphStorage,
        node_id: usize,
        layer: usize,
        m_max: usize,
        metric: DistanceMetric,
    ) {
        let current_neighbors: Vec<usize> = graph.get_neighbors(node_id, layer).to_vec();
        if current_neighbors.len() <= m_max {
            return;
        }

        let node_vec = vectors.get(node_id);

        // Compute distances to all neighbors
        let mut neighbor_dists: Vec<Candidate> = current_neighbors
            .iter()
            .map(|&nid| Candidate {
                id: nid,
                dist: compute_distance(node_vec, vectors.get(nid), metric),
            })
            .collect();

        // Use heuristic selection (diversity-aware with backfill)
        let selected = Self::select_neighbors_heuristic(vectors, neighbor_dists, m_max, metric);
        let selected_set: std::collections::HashSet<usize> =
            selected.iter().map(|c| c.id).collect();

        // Find pruned neighbors
        let pruned: Vec<usize> = current_neighbors
            .iter()
            .filter(|nid| !selected_set.contains(nid))
            .copied()
            .collect();

        // Update this node's neighbors
        let new_neighbors: Vec<usize> = selected.iter().map(|c| c.id).collect();
        graph.set_neighbors(node_id, layer, new_neighbors);

        // Remove bidirectional connections for pruned neighbors
        for &pruned_nid in &pruned {
            let nbs = graph.get_neighbors_mut(pruned_nid, layer);
            nbs.retain(|&id| id != node_id);
        }
    }

    // ---------------------------------------------------------------------
    // insert_node — graph construction
    // ---------------------------------------------------------------------

    fn insert_node(&self, new_id: usize, new_level: usize) {
        let vectors = self.vectors.read();
        let query = vectors.get(new_id);
        let entry_level = *self.entry_level.read();

        let ep = match *self.entry_point.read() {
            Some(ep) => ep,
            None => return,
        };

        let mut current = ep;
        let alive = self.alive.read();

        // Navigate upper layers (above new_level) greedily
        {
            let graph = self.graph.read();
            for lc in (new_level.saturating_add(1)..=entry_level).rev() {
                let mut changed = true;
                while changed {
                    changed = false;
                    for &nid in graph.get_neighbors(current, lc) {
                        if !alive[nid] { continue; }
                        if compute_distance(query, vectors.get(nid), self.metric)
                            < compute_distance(query, vectors.get(current), self.metric)
                        {
                            current = nid;
                            changed = true;
                        }
                    }
                }
            }
        }

        // Insert at layers min(new_level, entry_level) down to 0
        let top = new_level.min(entry_level);
        for lc in (0..=top).rev() {
            let candidates = {
                let graph = self.graph.read();
                Self::search_layer(&vectors, &graph, &alive, query, current, self.ef_construction, lc, self.metric)
            };

            if let Some(best) = candidates.iter().min_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap()) {
                current = best.id;
            }

            let m_max = self.m_max(lc);

            // Heuristic neighbor selection
            let selected = Self::select_neighbors_heuristic(&vectors, candidates, m_max, self.metric);
            let selected_ids: Vec<usize> = selected.iter().map(|c| c.id).collect();

            {
                let mut graph = self.graph.write();

                // Connect new node to selected neighbors
                graph.set_neighbors(new_id, lc, selected_ids.clone());

                // Add reverse connections and prune if needed
                for &neighbor_id in &selected_ids {
                    let nbs = graph.get_neighbors_mut(neighbor_id, lc);
                    nbs.push(new_id);
                    if nbs.len() > m_max {
                        Self::prune_connections(&vectors, &mut graph, neighbor_id, lc, m_max, self.metric);
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // Public API
    // ---------------------------------------------------------------------

    /// Add a vector. Returns its internal index.
    pub fn add_vector(&self, vector: &[f32]) -> usize {
        assert_eq!(vector.len(), self.dim, "Vector dimension mismatch");

        let level = self.random_level();

        let idx = {
            let mut vecs = self.vectors.write();
            vecs.add(vector)
        };

        {
            let mut graph = self.graph.write();
            graph.add_node(level);
        }

        {
            let mut alive = self.alive.write();
            if alive.len() <= idx {
                alive.resize(idx + 1, false);
            }
            alive[idx] = true;
        }

        let is_first = self.entry_point.read().is_none();
        if is_first {
            *self.entry_point.write() = Some(idx);
            *self.entry_level.write() = level;
            return idx;
        }

        self.insert_node(idx, level);

        if level > *self.entry_level.read() {
            *self.entry_point.write() = Some(idx);
            *self.entry_level.write() = level;
        }

        idx
    }

    /// Upsert a vector: update in-place if exists, insert if new.
    /// Returns (index, was_update). In-place update preserves the index
    /// and reconnects the node in the graph with fresh connections
    /// based on the new vector position.
    pub fn upsert_vector(&self, idx_if_exists: Option<usize>, vector: &[f32]) -> (usize, bool) {
        assert_eq!(vector.len(), self.dim, "Vector dimension mismatch");

        match idx_if_exists {
            Some(idx) => {
                // Check if actually alive
                let is_alive = {
                    let alive = self.alive.read();
                    idx < alive.len() && alive[idx]
                };

                if !is_alive {
                    // Dead slot — treat as fresh insert
                    return (self.add_vector(vector), false);
                }

                // --- In-place update ---

                // 1. Overwrite vector data
                self.vectors.write().set(idx, vector);

                // 2. Disconnect old graph edges (same as remove_vector)
                {
                    let graph = self.graph.read();
                    let level = graph.level(idx);
                    let mut to_clean: Vec<(usize, usize)> = Vec::new();
                    for lc in 0..=level {
                        for &nid in graph.get_neighbors(idx, lc) {
                            to_clean.push((nid, lc));
                        }
                    }
                    drop(graph);

                    let mut graph = self.graph.write();
                    for (nid, lc) in to_clean {
                        let nbs = graph.get_neighbors_mut(nid, lc);
                        nbs.retain(|&id| id != idx);
                    }
                    // Clear this node's neighbor lists
                    for lc in 0..=graph.level(idx) {
                        graph.set_neighbors(idx, lc, Vec::new());
                    }
                }

                // 3. Re-insert into graph with new connections
                let level = self.graph.read().level(idx);
                self.insert_node(idx, level);

                // 4. Update entry point if this node has a higher level
                if level > *self.entry_level.read() {
                    *self.entry_point.write() = Some(idx);
                    *self.entry_level.write() = level;
                }

                (idx, true)
            }
            None => {
                // New vector — normal insert
                (self.add_vector(vector), false)
            }
        }
    }

    /// Remove a vector by internal index.
    pub fn remove_vector(&self, idx: usize) -> bool {
        {
            let alive = self.alive.read();
            if idx >= alive.len() || !alive[idx] {
                return false;
            }
        }

        // Mark as dead
        self.alive.write()[idx] = false;

        // Remove from neighbor lists
        let graph = self.graph.read();
        let level = graph.level(idx);
        let mut to_clean: Vec<(usize, usize)> = Vec::new(); // (neighbor_id, layer)

        for lc in 0..=level {
            for &nid in graph.get_neighbors(idx, lc) {
                to_clean.push((nid, lc));
            }
        }
        drop(graph);

        // Remove bidirectional connections
        let mut graph = self.graph.write();
        for (nid, lc) in to_clean {
            let nbs = graph.get_neighbors_mut(nid, lc);
            nbs.retain(|&id| id != idx);
        }
        // Clear this node's own neighbor lists
        for lc in 0..=graph.level(idx) {
            graph.set_neighbors(idx, lc, Vec::new());
        }

        // Update entry point if needed
        let ep = *self.entry_point.read();
        if ep == Some(idx) {
            // Find a new entry point (any alive node)
            let alive = self.alive.read();
            let new_ep = alive.iter().enumerate().find(|(_, &a)| a).map(|(i, _)| i);
            *self.entry_point.write() = new_ep;
            if let Some(new_ep_id) = new_ep {
                *self.entry_level.write() = graph.level(new_ep_id);
            }
        }

        true
    }

    /// Search for k nearest neighbors.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(usize, f32)> {
        let ep = match *self.entry_point.read() {
            Some(ep) => ep,
            None => return Vec::new(),
        };

        let vectors = self.vectors.read();
        let graph = self.graph.read();
        let alive = self.alive.read();
        let entry_level = *self.entry_level.read();

        // Navigate upper layers greedily
        let mut current = ep;
        for lc in (1..=entry_level).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                for &nid in graph.get_neighbors(current, lc) {
                    if !alive[nid] { continue; }
                    if compute_distance(query, vectors.get(nid), self.metric)
                        < compute_distance(query, vectors.get(current), self.metric)
                    {
                        current = nid;
                        changed = true;
                    }
                }
            }
        }

        // Search layer 0
        let mut results = Self::search_layer(&vectors, &graph, &alive, query, current, ef, 0, self.metric);
        results.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        results.truncate(k);
        results.into_iter().map(|c| (c.id, c.dist)).collect()
    }

    /// Batch search with rayon parallelism.
    pub fn batch_search(
        &self,
        queries: &[&[f32]],
        k: usize,
        ef: usize,
    ) -> Vec<Vec<(usize, f32)>> {
        let ep = match *self.entry_point.read() {
            Some(ep) => ep,
            None => return queries.iter().map(|_| Vec::new()).collect(),
        };

        let vectors = self.vectors.read();
        let graph = self.graph.read();
        let alive = self.alive.read();
        let entry_level = *self.entry_level.read();
        let metric = self.metric;

        queries.par_iter().map(|query| {
            // Navigate upper layers
            let mut current = ep;
            for lc in (1..=entry_level).rev() {
                let mut changed = true;
                while changed {
                    changed = false;
                    for &nid in graph.get_neighbors(current, lc) {
                        if !alive[nid] { continue; }
                        if compute_distance(query, vectors.get(nid), metric)
                            < compute_distance(query, vectors.get(current), metric)
                        {
                            current = nid;
                            changed = true;
                        }
                    }
                }
            }

            let mut results = Self::search_layer(&vectors, &graph, &alive, query, current, ef, 0, metric);
            results.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
            results.truncate(k);
            results.into_iter().map(|c| (c.id, c.dist)).collect()
        }).collect()
    }

    /// Rebuild the entire index from stored vectors.
    pub fn rebuild(&self) {
        let old_vectors = self.vectors.read();
        let old_alive = self.alive.read();
        let n = old_vectors.len();
        let dim = old_vectors.dim();

        // Collect alive vectors
        let mut live_vecs: Vec<(usize, Vec<f32>)> = Vec::new();
        for i in 0..n {
            if i < old_alive.len() && old_alive[i] {
                live_vecs.push((i, old_vectors.get(i).to_vec()));
            }
        }
        drop(old_vectors);
        drop(old_alive);

        // Reset everything
        *self.graph.write() = GraphStorage::new(live_vecs.len());
        let mut new_vecs = VectorStorage::new(dim, live_vecs.len().max(1));
        let mut new_alive = Vec::new();

        *self.entry_point.write() = None;
        *self.entry_level.write() = 0;

        // Re-insert all vectors
        for (_, vec) in &live_vecs {
            new_vecs.add(vec);
            new_alive.push(true);
        }
        *self.vectors.write() = new_vecs;
        *self.alive.write() = new_alive;

        // Re-add nodes to graph
        let n_live = live_vecs.len();
        for _ in 0..n_live {
            let level = self.random_level();
            self.graph.write().add_node(level);
        }

        // Set first as entry point
        if n_live > 0 {
            *self.entry_point.write() = Some(0);
            *self.entry_level.write() = self.graph.read().level(0);
        }

        // Insert nodes 1..n into the graph
        for i in 1..n_live {
            let level = self.graph.read().level(i);
            self.insert_node(i, level);

            if level > *self.entry_level.read() {
                *self.entry_point.write() = Some(i);
                *self.entry_level.write() = level;
            }
        }
    }

    pub fn len(&self) -> usize {
        self.alive.read().iter().filter(|&&a| a).count()
    }

    pub fn total_allocated(&self) -> usize {
        self.vectors.read().len()
    }

    pub fn clear(&mut self) {
        *self.vectors.get_mut() = VectorStorage::new(self.dim, 1024);
        *self.graph.get_mut() = GraphStorage::new(1024);
        *self.entry_point.get_mut() = None;
        *self.entry_level.get_mut() = 0;
        self.alive.get_mut().clear();
    }

    pub fn dimension(&self) -> usize {
        self.dim
    }

    pub fn get_m(&self) -> usize {
        self.m
    }

    pub fn get_ef_construction(&self) -> usize {
        self.ef_construction
    }
}

// -------------------------------------------------------------------------
// PyO3 wrapper — String↔usize mapping happens HERE ONLY
// -------------------------------------------------------------------------

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use numpy::PyReadonlyArray1;
use std::collections::HashMap;

#[pyclass]
pub struct RustFastHNSWIndex {
    inner: FastHNSW,
    /// String ID → internal usize index
    id_to_idx: RwLock<HashMap<String, usize>>,
    /// Internal usize index → String ID
    idx_to_id: RwLock<Vec<String>>,
}

#[pymethods]
impl RustFastHNSWIndex {
    #[new]
    #[pyo3(signature = (dimension, m=16, ef_construction=200, ef_search=50, max_level=16, metric="cosine"))]
    fn new(
        dimension: usize,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        max_level: usize,
        metric: &str,
    ) -> PyResult<Self> {
        let dist_metric = match metric {
            "cosine" => DistanceMetric::Cosine,
            "l2" | "euclidean" => DistanceMetric::L2,
            "ip" | "inner_product" | "dot" => DistanceMetric::InnerProduct,
            _ => return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unknown metric '{}'. Use: cosine, l2, inner_product", metric)
            )),
        };
        Ok(Self {
            inner: FastHNSW::new(dimension, m, ef_construction, ef_search, max_level, dist_metric),
            id_to_idx: RwLock::new(HashMap::new()),
            idx_to_id: RwLock::new(Vec::new()),
        })
    }

    fn add_vector(&self, vector_id: String, vector: PyReadonlyArray1<f32>) -> PyResult<()> {
        let data = vector.as_slice()?;

        if data.len() != self.inner.dimension() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Vector dimension {} doesn't match index dimension {}",
                data.len(), self.inner.dimension()
            )));
        }

        // Check duplicate
        if self.id_to_idx.read().contains_key(&vector_id) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Vector ID {} already exists in index", vector_id
            )));
        }

        let idx = self.inner.add_vector(data);

        self.id_to_idx.write().insert(vector_id.clone(), idx);
        let mut ids = self.idx_to_id.write();
        if ids.len() <= idx {
            ids.resize(idx + 1, String::new());
        }
        ids[idx] = vector_id;

        Ok(())
    }

    fn remove_vector(&self, vector_id: String) -> bool {
        let idx = match self.id_to_idx.read().get(&vector_id) {
            Some(&idx) => idx,
            None => return false,
        };

        if self.inner.remove_vector(idx) {
            self.id_to_idx.write().remove(&vector_id);
            // Don't remove from idx_to_id (it's index-based), just clear it
            let mut ids = self.idx_to_id.write();
            if idx < ids.len() {
                ids[idx] = String::new();
            }
            true
        } else {
            false
        }
    }

    /// Upsert: insert or update a vector. Returns true if it was an update.
    /// If the ID exists, overwrites the vector data in-place and reconnects
    /// the node in the graph. If new, performs a normal insert.
    fn upsert_vector(&self, vector_id: String, vector: PyReadonlyArray1<f32>) -> PyResult<bool> {
        let data = vector.as_slice()?;

        if data.len() != self.inner.dimension() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Vector dimension {} doesn't match index dimension {}",
                data.len(), self.inner.dimension()
            )));
        }

        let existing_idx = self.id_to_idx.read().get(&vector_id).copied();

        let (idx, was_update) = self.inner.upsert_vector(existing_idx, data);

        if !was_update {
            // New insert — register the mapping
            self.id_to_idx.write().insert(vector_id.clone(), idx);
            let mut ids = self.idx_to_id.write();
            if ids.len() <= idx {
                ids.resize(idx + 1, String::new());
            }
            ids[idx] = vector_id;
        }
        // If was_update, mappings are unchanged (same idx, same ID)

        Ok(was_update)
    }

    #[pyo3(signature = (query_vector, k, distance_threshold=None, ef_override=None))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        query_vector: PyReadonlyArray1<f32>,
        k: usize,
        distance_threshold: Option<f32>,
        ef_override: Option<usize>,
    ) -> PyResult<Bound<'py, PyList>> {
        let query = query_vector.as_slice()?;

        if query.len() != self.inner.dimension() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Query dimension {} doesn't match index dimension {}",
                query.len(), self.inner.dimension()
            )));
        }

        let ef = ef_override.unwrap_or(self.inner.ef_search);
        let results = self.inner.search(query, k, ef);

        let idx_to_id = self.idx_to_id.read();
        let result_list = PyList::empty(py);
        for (idx, dist) in results {
            if let Some(threshold) = distance_threshold {
                if dist > threshold { continue; }
            }
            if idx < idx_to_id.len() && !idx_to_id[idx].is_empty() {
                result_list.append((idx_to_id[idx].as_str(), dist).into_pyobject(py).unwrap().into_any().unbind())?;
            }
        }
        Ok(result_list)
    }

    /// Pre-filtered search: only return results whose IDs are in filter_ids.
    /// Oversamples by 2x ef to compensate for filtered-out candidates.
    #[pyo3(signature = (query_vector, k, filter_ids, distance_threshold=None, ef_override=None))]
    fn search_filtered<'py>(
        &self,
        py: Python<'py>,
        query_vector: PyReadonlyArray1<f32>,
        k: usize,
        filter_ids: Vec<String>,
        distance_threshold: Option<f32>,
        ef_override: Option<usize>,
    ) -> PyResult<Bound<'py, PyList>> {
        let query = query_vector.as_slice()?;
        let ef = ef_override.unwrap_or(self.inner.ef_search);

        // Build a bitset of allowed internal indices
        let id_to_idx = self.id_to_idx.read();
        let n = self.inner.total_allocated();
        let mut allowed = vec![false; n];
        for fid in &filter_ids {
            if let Some(&idx) = id_to_idx.get(fid) {
                allowed[idx] = true;
            }
        }

        // Oversample to compensate for filtering
        let oversample_ef = (ef * 2).max(k * 4);
        let results = self.inner.search(query, oversample_ef, oversample_ef);

        let idx_to_id = self.idx_to_id.read();
        let result_list = PyList::empty(py);
        let mut count = 0;
        for (idx, dist) in results {
            if count >= k { break; }
            if idx >= allowed.len() || !allowed[idx] { continue; }
            if let Some(threshold) = distance_threshold {
                if dist > threshold { continue; }
            }
            if idx < idx_to_id.len() && !idx_to_id[idx].is_empty() {
                result_list.append((idx_to_id[idx].as_str(), dist).into_pyobject(py).unwrap().into_any().unbind())?;
                count += 1;
            }
        }
        Ok(result_list)
    }

    fn set_ef_search(&mut self, ef_search: usize) {
        self.inner.ef_search = ef_search;
    }

    fn get_ef_search(&self) -> usize {
        self.inner.ef_search
    }

    fn size(&self) -> usize {
        self.inner.len()
    }

    fn clear(&mut self) {
        self.inner.clear();
        self.id_to_idx.get_mut().clear();
        self.idx_to_id.get_mut().clear();
    }

    fn rebuild(&self) -> PyResult<()> {
        self.inner.rebuild();
        Ok(())
    }

    fn get_statistics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stats = PyDict::new(py);
        let size = self.inner.len();
        let total = self.inner.total_allocated();
        stats.set_item("size", size)?;
        stats.set_item("total_allocated", total)?;
        stats.set_item("dimension", self.inner.dimension())?;
        stats.set_item("M", self.inner.get_m())?;
        stats.set_item("ef_construction", self.inner.get_ef_construction())?;
        stats.set_item("ef_search", self.inner.ef_search)?;
        stats.set_item("max_level", self.inner.max_level)?;

        // Compute actual max level from graph
        let graph = self.inner.graph.read();
        let alive = self.inner.alive.read();
        let mut actual_max = 0usize;
        for i in 0..graph.len() {
            if i < alive.len() && alive[i] {
                actual_max = actual_max.max(graph.level(i));
            }
        }
        stats.set_item("current_max_level", actual_max)?;

        Ok(stats)
    }

    /// Batch search with rayon parallelism.
    fn batch_search<'py>(
        &self,
        py: Python<'py>,
        query_vectors: &Bound<'py, PyList>,
        k: usize,
        distance_threshold: Option<f32>,
    ) -> PyResult<Bound<'py, PyList>> {
        let mut queries: Vec<Vec<f32>> = Vec::new();
        for item in query_vectors.iter() {
            let arr: PyReadonlyArray1<f32> = item.extract()?;
            queries.push(arr.as_slice()?.to_vec());
        }

        let query_refs: Vec<&[f32]> = queries.iter().map(|q| q.as_slice()).collect();
        let ef = self.inner.ef_search;
        let batch_results = self.inner.batch_search(&query_refs, k, ef);

        let idx_to_id = self.idx_to_id.read();
        let outer = PyList::empty(py);

        for results in batch_results {
            let inner = PyList::empty(py);
            for (idx, dist) in results {
                if let Some(threshold) = distance_threshold {
                    if dist > threshold { continue; }
                }
                if idx < idx_to_id.len() && !idx_to_id[idx].is_empty() {
                    inner.append((idx_to_id[idx].as_str(), dist).into_pyobject(py).unwrap().into_any().unbind())?;
                }
            }
            outer.append(inner)?;
        }

        Ok(outer)
    }
}
