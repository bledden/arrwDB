use std::collections::{HashMap, HashSet};

/// Node in the HNSW graph.
///
/// Each node represents a vector and maintains connections to neighbors
/// at multiple layers of the hierarchy.
#[derive(Debug, Clone)]
pub struct HNSWNode {
    /// Unique identifier for the vector
    pub vector_id: String,
    /// Maximum level this node appears in
    pub level: usize,
    /// neighbors[layer] = set of neighbor vector_ids at that layer
    pub neighbors: HashMap<usize, HashSet<String>>,
}
