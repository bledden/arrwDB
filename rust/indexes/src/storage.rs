/// Contiguous storage for vectors and graph structure.
/// All lookups are O(1) array indexing — no HashMap, no String hashing.

/// Flat vector storage: vectors are stored contiguously for cache-friendly access.
/// `get(id)` is a single pointer offset, not a HashMap lookup.
pub struct VectorStorage {
    /// Flat vector data: vectors[id * dim .. (id+1) * dim]
    data: Vec<f32>,
    dim: usize,
    count: usize,
}

impl VectorStorage {
    pub fn new(dim: usize, capacity: usize) -> Self {
        Self {
            data: vec![0.0f32; dim * capacity],
            dim,
            count: 0,
        }
    }

    /// Add a vector, returns its index. O(1) amortized.
    pub fn add(&mut self, vector: &[f32]) -> usize {
        debug_assert_eq!(vector.len(), self.dim);
        let idx = self.count;
        let needed = (idx + 1) * self.dim;
        if needed > self.data.len() {
            let new_cap = (self.data.len() / self.dim * 3 / 2 + 1) * self.dim;
            self.data.resize(new_cap.max(needed), 0.0f32);
        }
        let start = idx * self.dim;
        self.data[start..start + self.dim].copy_from_slice(vector);
        self.count += 1;
        idx
    }

    /// Get vector by index — zero-copy slice. O(1).
    #[inline(always)]
    pub fn get(&self, idx: usize) -> &[f32] {
        let start = idx * self.dim;
        &self.data[start..start + self.dim]
    }

    /// Overwrite vector at index (used by rebuild).
    pub fn set(&mut self, idx: usize, vector: &[f32]) {
        debug_assert_eq!(vector.len(), self.dim);
        let start = idx * self.dim;
        self.data[start..start + self.dim].copy_from_slice(vector);
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Reset to empty (keeps allocated memory).
    pub fn clear(&mut self) {
        self.count = 0;
    }
}

/// Graph storage: flat arrays of neighbor lists per layer.
pub struct GraphStorage {
    /// neighbors[node_id].layers[layer] = Vec<usize> of neighbor indices
    nodes: Vec<GraphNode>,
}

pub struct GraphNode {
    /// Level assigned to this node (max layer it appears in)
    pub level: usize,
    /// Per-layer neighbor lists. layers[0] is layer 0 (densest), etc.
    pub layers: Vec<Vec<usize>>,
}

impl GraphStorage {
    pub fn new(capacity: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(capacity),
        }
    }

    /// Add a node with the given level. Returns node index.
    pub fn add_node(&mut self, level: usize) -> usize {
        let idx = self.nodes.len();
        let mut layers = Vec::with_capacity(level + 1);
        for _ in 0..=level {
            layers.push(Vec::new());
        }
        self.nodes.push(GraphNode { level, layers });
        idx
    }

    #[inline(always)]
    pub fn get_neighbors(&self, node_id: usize, layer: usize) -> &[usize] {
        &self.nodes[node_id].layers[layer]
    }

    #[inline(always)]
    pub fn get_neighbors_mut(&mut self, node_id: usize, layer: usize) -> &mut Vec<usize> {
        &mut self.nodes[node_id].layers[layer]
    }

    /// Set neighbors for a node at a layer (replaces existing).
    pub fn set_neighbors(&mut self, node_id: usize, layer: usize, neighbors: Vec<usize>) {
        self.nodes[node_id].layers[layer] = neighbors;
    }

    #[inline(always)]
    pub fn level(&self, node_id: usize) -> usize {
        self.nodes[node_id].level
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Reset to empty (keeps allocated memory).
    pub fn clear(&mut self) {
        self.nodes.clear();
    }
}
