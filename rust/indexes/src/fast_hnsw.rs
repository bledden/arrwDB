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

use crate::distance::{compute_distance, prefetch_vector, DistanceMetric};
use crate::storage::{GraphStorage, VectorStorage, VectorStore};
use parking_lot::RwLock;
use rand::Rng;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::cell::RefCell;

// -------------------------------------------------------------------------
// Visited list with generation counter (hnswlib technique)
// O(1) reset by incrementing generation. No allocation per query.
// Full memset only every 65,535 searches.
// -------------------------------------------------------------------------

pub struct VisitedList {
    generation: u16,
    marks: Vec<u16>,
}

impl VisitedList {
    fn new(capacity: usize) -> Self {
        Self {
            generation: 1,
            marks: vec![0u16; capacity],
        }
    }

    pub fn ensure_capacity(&mut self, capacity: usize) {
        if self.marks.len() < capacity {
            self.marks.resize(capacity, 0);
        }
    }

    /// O(1) reset — just bump the generation counter.
    /// Full memset only on u16 overflow (every 65,535 resets).
    pub fn reset(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.marks.fill(0);
            self.generation = 1;
        }
    }

    #[inline(always)]
    pub fn is_visited(&self, id: usize) -> bool {
        unsafe { *self.marks.get_unchecked(id) == self.generation }
    }

    #[inline(always)]
    pub fn mark_visited(&mut self, id: usize) {
        unsafe { *self.marks.get_unchecked_mut(id) = self.generation; }
    }
}

// Thread-local visited list pool — zero allocation in the search hot path
thread_local! {
    static VISITED_LIST: RefCell<VisitedList> = RefCell::new(VisitedList::new(0));
}

// -------------------------------------------------------------------------
// Candidate types for BinaryHeap
// -------------------------------------------------------------------------

#[derive(Clone)]
pub struct Candidate {
    pub id: usize,
    pub dist: f32,
}

/// Min-heap wrapper (smallest distance = highest priority).
pub struct MinCand(pub Candidate);

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
pub struct MaxCand(pub Candidate);

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

    pub vectors: RwLock<VectorStore>,
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
            vectors: RwLock::new(VectorStore::InMemory(VectorStorage::new(dim, 1024))),
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
        vectors: &VectorStore,
        graph: &GraphStorage,
        alive: &[bool],
        query: &[f32],
        entry_id: usize,
        ef: usize,
        layer: usize,
        metric: DistanceMetric,
    ) -> Vec<Candidate> {
        let n = graph.len();

        // Use thread-local visited list with generation counter.
        // O(1) reset, zero allocation per query.
        VISITED_LIST.with(|vl| {
            let mut visited = vl.borrow_mut();
            visited.ensure_capacity(n);
            visited.reset();
            visited.mark_visited(entry_id);
            Self::_search_layer_inner(vectors, graph, alive, &mut visited, query, entry_id, ef, layer, metric)
        })
    }

    /// Inner search loop factored out so the visited list borrow is clear.
    fn _search_layer_inner(
        vectors: &VectorStore,
        graph: &GraphStorage,
        alive: &[bool],
        visited: &mut VisitedList,
        query: &[f32],
        entry_id: usize,
        ef: usize,
        layer: usize,
        metric: DistanceMetric,
    ) -> Vec<Candidate> {

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
            for (ni, &nid) in neighbors.iter().enumerate() {
                if visited.is_visited(nid) || !alive[nid] {
                    continue;
                }
                visited.mark_visited(nid);

                // Prefetch the NEXT neighbor's vector while computing this one
                if ni + 1 < neighbors.len() {
                    let next_nid = neighbors[ni + 1];
                    if !visited.is_visited(next_nid) {
                        prefetch_vector(vectors.get(next_nid).as_ptr());
                    }
                }

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

    /// Filtered search at a single layer.
    /// Traverses all neighbors (for graph connectivity) but only adds
    /// allowed nodes to results. This is the Qdrant-style approach.
    fn search_layer_filtered(
        vectors: &VectorStore,
        graph: &GraphStorage,
        alive: &[bool],
        allowed: &[bool],
        query: &[f32],
        entry_id: usize,
        ef: usize,
        layer: usize,
        metric: DistanceMetric,
    ) -> Vec<Candidate> {
        let n = graph.len();

        VISITED_LIST.with(|vl| {
            let mut visited = vl.borrow_mut();
            visited.ensure_capacity(n);
            visited.reset();
            visited.mark_visited(entry_id);
            Self::_search_layer_filtered_inner(vectors, graph, alive, allowed, &mut visited, query, entry_id, ef, layer, metric)
        })
    }

    fn _search_layer_filtered_inner(
        vectors: &VectorStore,
        graph: &GraphStorage,
        alive: &[bool],
        allowed: &[bool],
        visited: &mut VisitedList,
        query: &[f32],
        entry_id: usize,
        ef: usize,
        layer: usize,
        metric: DistanceMetric,
    ) -> Vec<Candidate> {
        let entry_dist = compute_distance(query, vectors.get(entry_id), metric);

        let mut candidates = BinaryHeap::new();
        candidates.push(MinCand(Candidate { id: entry_id, dist: entry_dist }));

        let mut results = BinaryHeap::new();
        if entry_id < allowed.len() && allowed[entry_id] {
            results.push(MaxCand(Candidate { id: entry_id, dist: entry_dist }));
        }

        while let Some(MinCand(current)) = candidates.pop() {
            // Use worst filtered result for pruning, but be generous
            let worst_dist = if results.is_empty() {
                f32::MAX
            } else {
                results.peek().unwrap().0.dist
            };

            let neighbors = graph.get_neighbors(current.id, layer);
            for (ni, &nid) in neighbors.iter().enumerate() {
                if visited.is_visited(nid) || !alive[nid] {
                    continue;
                }
                visited.mark_visited(nid);

                if ni + 1 < neighbors.len() {
                    let next_nid = neighbors[ni + 1];
                    if !visited.is_visited(next_nid) {
                        prefetch_vector(vectors.get(next_nid).as_ptr());
                    }
                }

                let dist = compute_distance(query, vectors.get(nid), metric);

                // ALWAYS add to candidates (for graph traversal)
                if dist < worst_dist || results.len() < ef {
                    candidates.push(MinCand(Candidate { id: nid, dist }));
                }

                // Only add to results if it passes the filter
                if nid < allowed.len() && allowed[nid] {
                    if dist < worst_dist || results.len() < ef {
                        results.push(MaxCand(Candidate { id: nid, dist }));
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }

            // Termination: all remaining candidates are worse than worst filtered result
            if let Some(next) = candidates.peek() {
                if !results.is_empty() && next.0.dist > results.peek().unwrap().0.dist && results.len() >= ef {
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
        vectors: &VectorStore,
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
        vectors: &VectorStore,
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

    /// Batch-add all vectors at once. Much faster than individual add_vector calls
    /// because: (1) single lock acquisition for all vectors, (2) pre-allocated storage,
    /// (3) no PyO3 boundary crossing per vector.
    /// Returns the index of the first added vector.
    pub fn batch_add_vectors(&self, vectors: &[f32], n: usize) -> usize {
        assert_eq!(vectors.len(), n * self.dim);

        let first_idx;

        // 1. Pre-allocate and add all vectors + graph nodes in bulk
        {
            let mut vecs = self.vectors.write();
            first_idx = vecs.len();
            for i in 0..n {
                let start = i * self.dim;
                vecs.add(&vectors[start..start + self.dim]);
            }
        }

        {
            let mut graph = self.graph.write();
            for _ in 0..n {
                let level = self.random_level();
                graph.add_node(level);
            }
        }

        {
            let mut alive = self.alive.write();
            alive.resize(first_idx + n, true);
        }

        // 2. Set first node as entry point
        if self.entry_point.read().is_none() {
            *self.entry_point.write() = Some(first_idx);
            *self.entry_level.write() = self.graph.read().level(first_idx);
        }

        // 3. Insert each node into the graph
        for i in 0..n {
            let idx = first_idx + i;
            if idx == first_idx && self.graph.read().len() == n {
                // First node in empty graph — skip insert_node (it's the entry point)
                continue;
            }
            let level = self.graph.read().level(idx);
            self.insert_node(idx, level);

            if level > *self.entry_level.read() {
                *self.entry_point.write() = Some(idx);
                *self.entry_level.write() = level;
            }

            // Progress reporting for large batches
            if (i + 1) % 50000 == 0 {
                // Can't log from Rust easily, but the caller can check size()
            }
        }

        first_idx
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

    /// Search with graph-integrated filtering.
    /// Non-matching nodes are still traversed (for graph connectivity)
    /// but only matching nodes appear in results.
    pub fn search_filtered(&self, query: &[f32], k: usize, ef: usize, allowed: &[bool]) -> Vec<(usize, f32)> {
        let ep = match *self.entry_point.read() {
            Some(ep) => ep,
            None => return Vec::new(),
        };

        let vectors = self.vectors.read();
        let graph = self.graph.read();
        let alive = self.alive.read();
        let entry_level = *self.entry_level.read();

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

        let mut results = Self::search_layer_filtered(
            &vectors, &graph, &alive, allowed, query, current, ef, 0, self.metric,
        );
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
        let mut new_vecs = VectorStore::InMemory(VectorStorage::new(dim, live_vecs.len().max(1)));
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
        *self.vectors.get_mut() = VectorStore::InMemory(VectorStorage::new(self.dim, 1024));
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
            "manhattan" | "l1" => DistanceMetric::Manhattan,
            "hamming" => DistanceMetric::Hamming,
            _ => return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unknown metric '{}'. Use: cosine, l2, inner_product, manhattan, hamming", metric)
            )),
        };
        Ok(Self {
            inner: FastHNSW::new(dimension, m, ef_construction, ef_search, max_level, dist_metric),
            id_to_idx: RwLock::new(HashMap::new()),
            idx_to_id: RwLock::new(Vec::new()),
        })
    }

    /// Batch-add vectors from a flat numpy array + list of IDs.
    /// Much faster than calling add_vector() in a Python loop.
    fn batch_add_vectors(
        &self,
        vector_ids: Vec<String>,
        vectors_flat: PyReadonlyArray1<f32>,
    ) -> PyResult<()> {
        let data = vectors_flat.as_slice()?;
        let n = vector_ids.len();
        let dim = self.inner.dimension();

        if data.len() != n * dim {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected {} floats ({}x{}), got {}",
                n * dim, n, dim, data.len()
            )));
        }

        // Check for duplicates
        {
            let id_map = self.id_to_idx.read();
            for vid in &vector_ids {
                if id_map.contains_key(vid) {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        format!("Vector ID {} already exists", vid)
                    ));
                }
            }
        }

        let first_idx = self.inner.batch_add_vectors(data, n);

        // Register all ID mappings
        let mut id_map = self.id_to_idx.write();
        let mut ids = self.idx_to_id.write();
        ids.resize(first_idx + n, String::new());

        for (i, vid) in vector_ids.into_iter().enumerate() {
            let idx = first_idx + i;
            id_map.insert(vid.clone(), idx);
            ids[idx] = vid;
        }

        Ok(())
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

    /// Pre-filtered search: graph-integrated filtering.
    /// Non-matching nodes are still traversed (for graph connectivity) but only
    /// matching nodes appear in results. No oversampling waste.
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

        // Graph-integrated filtering: traverse all nodes, only return matching
        let results = self.inner.search_filtered(query, k, ef, &allowed);

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

    /// Save vector data to disk and switch to memory-mapped access.
    /// After this call, vectors are read from disk via OS page cache.
    /// The graph stays in RAM. Writes (add_vector) are disabled.
    fn persist_to_disk(&self, path: String) -> PyResult<()> {
        let mut vectors = self.inner.vectors.write();
        let store = std::mem::replace(
            &mut *vectors,
            VectorStore::InMemory(VectorStorage::new(self.inner.dim, 0)),
        );
        match store.persist_and_mmap(std::path::Path::new(&path)) {
            Ok(mmap_store) => {
                *vectors = mmap_store;
                Ok(())
            }
            Err(e) => Err(pyo3::exceptions::PyIOError::new_err(
                format!("Persist failed: {}", e)
            )),
        }
    }

    /// Check if the index is using disk-backed storage.
    fn is_disk_backed(&self) -> bool {
        matches!(&*self.inner.vectors.read(), VectorStore::Mmap(_))
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
