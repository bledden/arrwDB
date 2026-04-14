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

    /// Save vector data to a file for memory-mapped access.
    pub fn save_to_file(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;
        // Header: dim (8 bytes) + count (8 bytes)
        file.write_all(&(self.dim as u64).to_le_bytes())?;
        file.write_all(&(self.count as u64).to_le_bytes())?;
        // Vector data as raw f32 bytes
        let byte_slice = unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const u8,
                self.count * self.dim * std::mem::size_of::<f32>(),
            )
        };
        file.write_all(byte_slice)?;
        file.flush()?;
        Ok(())
    }

    /// Load from a memory-mapped file. Returns a MmapVectorStorage
    /// that can be used in place of VectorStorage for read-only search.
    pub fn load_mmap(path: &std::path::Path) -> std::io::Result<MmapVectorStorage> {
        MmapVectorStorage::open(path)
    }
}

/// Memory-mapped vector storage for disk-based access.
/// Vectors are read directly from disk via OS page cache — no RAM allocation
/// needed for the vector data itself. The OS transparently pages data in/out.
///
/// Read-only after creation. For writes, use VectorStorage then save_to_file().
pub struct MmapVectorStorage {
    mmap: memmap2::Mmap,
    dim: usize,
    count: usize,
}

impl MmapVectorStorage {
    /// Open a memory-mapped vector file created by VectorStorage::save_to_file().
    pub fn open(path: &std::path::Path) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };

        // Read header
        let dim = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        let count = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;

        Ok(Self { mmap, dim, count })
    }

    /// Get vector by index — zero-copy from mmap. O(1).
    #[inline(always)]
    pub fn get(&self, idx: usize) -> &[f32] {
        let header_size = 16; // 8 bytes dim + 8 bytes count
        let byte_offset = header_size + idx * self.dim * std::mem::size_of::<f32>();
        let byte_len = self.dim * std::mem::size_of::<f32>();
        let bytes = &self.mmap[byte_offset..byte_offset + byte_len];
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, self.dim) }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Enum storage: either in-memory or memory-mapped from disk.
/// No trait object indirection — the match is branch-predicted after first call.
pub enum VectorStore {
    InMemory(VectorStorage),
    Mmap(MmapVectorStorage),
}

impl VectorStore {
    #[inline(always)]
    pub fn get(&self, idx: usize) -> &[f32] {
        match self {
            VectorStore::InMemory(s) => s.get(idx),
            VectorStore::Mmap(s) => s.get(idx),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        match self {
            VectorStore::InMemory(s) => s.len(),
            VectorStore::Mmap(s) => s.len(),
        }
    }

    #[inline]
    pub fn dim(&self) -> usize {
        match self {
            VectorStore::InMemory(s) => s.dim(),
            VectorStore::Mmap(s) => s.dim(),
        }
    }

    /// Add a vector (only works for in-memory mode).
    pub fn add(&mut self, vector: &[f32]) -> usize {
        match self {
            VectorStore::InMemory(s) => s.add(vector),
            VectorStore::Mmap(_) => panic!("Cannot add vectors to mmap storage (read-only)"),
        }
    }

    /// Overwrite a vector (only works for in-memory mode).
    pub fn set(&mut self, idx: usize, vector: &[f32]) {
        match self {
            VectorStore::InMemory(s) => s.set(idx, vector),
            VectorStore::Mmap(_) => panic!("Cannot set vectors in mmap storage (read-only)"),
        }
    }

    pub fn save_to_file(&self, path: &std::path::Path) -> std::io::Result<()> {
        match self {
            VectorStore::InMemory(s) => s.save_to_file(path),
            VectorStore::Mmap(_) => Ok(()), // Already on disk
        }
    }

    /// Convert from in-memory to disk-backed mmap.
    pub fn persist_and_mmap(self, path: &std::path::Path) -> std::io::Result<Self> {
        match &self {
            VectorStore::InMemory(s) => {
                s.save_to_file(path)?;
                let mmap = MmapVectorStorage::open(path)?;
                Ok(VectorStore::Mmap(mmap))
            }
            VectorStore::Mmap(_) => Ok(self), // Already mmap
        }
    }

    pub fn clear(&mut self) {
        match self {
            VectorStore::InMemory(s) => s.clear(),
            VectorStore::Mmap(_) => panic!("Cannot clear mmap storage"),
        }
    }
}

// -------------------------------------------------------------------------
// Co-located node storage: neighbors + vector in same cache lines
// This is the hnswlib technique that gives 1.4-1.6x speedup.
// -------------------------------------------------------------------------

/// Co-located storage for layer-0: each node's neighbor list and vector data
/// are adjacent in a single contiguous byte array. Prefetching a neighbor
/// brings in both its neighbor list AND vector data.
///
/// Memory layout per node (at offset `id * stride`):
///   [neighbor_count: u16][neighbors: u32 * m_max0][vector: f32 * dim]
///
/// For M=16 (m_max0=32), dim=128: stride = 2 + 128 + 512 = 642 bytes
pub struct ColocatedStore {
    data: Vec<u8>,
    stride: usize,
    dim: usize,
    m_max0: usize,
    count: usize,
    /// Offset from node start to neighbor data
    neighbors_offset: usize,
    /// Offset from node start to vector data
    vector_offset: usize,
}

impl ColocatedStore {
    pub fn new(dim: usize, m_max0: usize, capacity: usize) -> Self {
        let neighbors_offset = 2; // u16 count
        let vector_offset = neighbors_offset + m_max0 * 4; // u32 per neighbor
        let stride = vector_offset + dim * 4; // f32 per dimension
        Self {
            data: vec![0u8; stride * capacity],
            stride,
            dim,
            m_max0,
            count: 0,
            neighbors_offset,
            vector_offset,
        }
    }

    /// Add a node with its vector. Returns its index.
    pub fn add(&mut self, vector: &[f32]) -> usize {
        debug_assert_eq!(vector.len(), self.dim);
        let idx = self.count;
        let needed = (idx + 1) * self.stride;
        if needed > self.data.len() {
            let new_cap = (self.data.len() / self.stride * 3 / 2 + 1) * self.stride;
            self.data.resize(new_cap.max(needed), 0u8);
        }
        // Write vector data
        let vec_start = idx * self.stride + self.vector_offset;
        let vec_bytes = unsafe {
            std::slice::from_raw_parts(vector.as_ptr() as *const u8, self.dim * 4)
        };
        self.data[vec_start..vec_start + self.dim * 4].copy_from_slice(vec_bytes);
        // Neighbor count starts at 0
        self.set_neighbor_count(idx, 0);
        self.count += 1;
        idx
    }

    /// Get vector by index — zero-copy slice from co-located data.
    #[inline(always)]
    pub fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.stride + self.vector_offset;
        unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr().add(start) as *const f32,
                self.dim,
            )
        }
    }

    /// Overwrite vector at index.
    pub fn set_vector(&mut self, idx: usize, vector: &[f32]) {
        let start = idx * self.stride + self.vector_offset;
        let bytes = unsafe {
            std::slice::from_raw_parts(vector.as_ptr() as *const u8, self.dim * 4)
        };
        self.data[start..start + self.dim * 4].copy_from_slice(bytes);
    }

    /// Get neighbor count for a node.
    #[inline(always)]
    pub fn neighbor_count(&self, idx: usize) -> usize {
        let offset = idx * self.stride;
        u16::from_le_bytes([self.data[offset], self.data[offset + 1]]) as usize
    }

    /// Set neighbor count.
    fn set_neighbor_count(&mut self, idx: usize, count: usize) {
        let offset = idx * self.stride;
        let bytes = (count as u16).to_le_bytes();
        self.data[offset] = bytes[0];
        self.data[offset + 1] = bytes[1];
    }

    /// Get layer-0 neighbors as u32 slice.
    #[inline(always)]
    pub fn get_neighbors(&self, idx: usize) -> &[u32] {
        let count = self.neighbor_count(idx);
        let start = idx * self.stride + self.neighbors_offset;
        unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr().add(start) as *const u32,
                count,
            )
        }
    }

    /// Set layer-0 neighbors.
    pub fn set_neighbors(&mut self, idx: usize, neighbors: &[u32]) {
        debug_assert!(neighbors.len() <= self.m_max0);
        self.set_neighbor_count(idx, neighbors.len());
        let start = idx * self.stride + self.neighbors_offset;
        let bytes = unsafe {
            std::slice::from_raw_parts(
                neighbors.as_ptr() as *const u8,
                neighbors.len() * 4,
            )
        };
        self.data[start..start + bytes.len()].copy_from_slice(bytes);
    }

    /// Push a neighbor to a node's list.
    pub fn push_neighbor(&mut self, idx: usize, neighbor: u32) {
        let count = self.neighbor_count(idx);
        debug_assert!(count < self.m_max0);
        let start = idx * self.stride + self.neighbors_offset + count * 4;
        let bytes = neighbor.to_le_bytes();
        self.data[start..start + 4].copy_from_slice(&bytes);
        self.set_neighbor_count(idx, count + 1);
    }

    /// Remove a specific neighbor from a node's list.
    pub fn remove_neighbor(&mut self, idx: usize, neighbor: u32) {
        let count = self.neighbor_count(idx);
        let start = idx * self.stride + self.neighbors_offset;
        let mut new_count = 0;
        for i in 0..count {
            let offset = start + i * 4;
            let nid = u32::from_le_bytes([
                self.data[offset], self.data[offset + 1],
                self.data[offset + 2], self.data[offset + 3],
            ]);
            if nid != neighbor {
                if new_count != i {
                    // Shift left
                    let dst = start + new_count * 4;
                    self.data.copy_within(offset..offset + 4, dst);
                }
                new_count += 1;
            }
        }
        self.set_neighbor_count(idx, new_count);
    }

    /// Get raw pointer to a node's data (for prefetching).
    #[inline(always)]
    pub fn node_ptr(&self, idx: usize) -> *const u8 {
        unsafe { self.data.as_ptr().add(idx * self.stride) }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

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
