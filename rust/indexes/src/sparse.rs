/// Sparse vector storage and search.
///
/// Supports SPLADE, BM25 term weights, learned sparse representations,
/// and any other sparse embedding format.
///
/// Storage: compressed sparse row format (indices + values per vector).
/// Search: sparse dot product with inverted index for fast lookup.

use std::collections::HashMap;
use parking_lot::RwLock;
use pyo3::prelude::*;
use pyo3::types::PyList;

/// A single sparse vector: sorted list of (dimension_index, value) pairs.
#[derive(Clone)]
pub struct SparseVector {
    /// Sorted by index for efficient dot product
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

impl SparseVector {
    pub fn new(indices: Vec<u32>, values: Vec<f32>) -> Self {
        debug_assert_eq!(indices.len(), values.len());
        Self { indices, values }
    }

    /// Dot product between two sparse vectors.
    pub fn dot(&self, other: &SparseVector) -> f32 {
        let mut result = 0.0f32;
        let mut i = 0;
        let mut j = 0;
        while i < self.indices.len() && j < other.indices.len() {
            if self.indices[i] == other.indices[j] {
                result += self.values[i] * other.values[j];
                i += 1;
                j += 1;
            } else if self.indices[i] < other.indices[j] {
                i += 1;
            } else {
                j += 1;
            }
        }
        result
    }

    pub fn nnz(&self) -> usize {
        self.indices.len()
    }
}

/// Sparse vector index with inverted posting lists for fast search.
pub struct SparseIndex {
    /// All stored sparse vectors
    vectors: Vec<SparseVector>,
    /// Inverted index: dimension -> list of (vector_id, value)
    postings: HashMap<u32, Vec<(usize, f32)>>,
    /// Tracks alive vectors (for deletion)
    alive: Vec<bool>,
}

impl SparseIndex {
    pub fn new() -> Self {
        Self {
            vectors: Vec::new(),
            postings: HashMap::new(),
            alive: Vec::new(),
        }
    }

    /// Add a sparse vector. Returns its index.
    pub fn add(&mut self, vec: SparseVector) -> usize {
        let idx = self.vectors.len();
        // Update inverted index
        for (&dim_idx, &val) in vec.indices.iter().zip(vec.values.iter()) {
            self.postings
                .entry(dim_idx)
                .or_insert_with(Vec::new)
                .push((idx, val));
        }
        self.vectors.push(vec);
        self.alive.push(true);
        idx
    }

    /// Search for top-k vectors by sparse dot product.
    pub fn search(&self, query: &SparseVector, k: usize) -> Vec<(usize, f32)> {
        let mut scores: HashMap<usize, f32> = HashMap::new();

        // Accumulate scores via inverted index (only touches relevant dimensions)
        for (&q_idx, &q_val) in query.indices.iter().zip(query.values.iter()) {
            if let Some(posting) = self.postings.get(&q_idx) {
                for &(doc_id, doc_val) in posting {
                    if self.alive[doc_id] {
                        *scores.entry(doc_id).or_insert(0.0) += q_val * doc_val;
                    }
                }
            }
        }

        let mut results: Vec<(usize, f32)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    pub fn remove(&mut self, idx: usize) -> bool {
        if idx < self.alive.len() && self.alive[idx] {
            self.alive[idx] = false;
            true
        } else {
            false
        }
    }

    pub fn len(&self) -> usize {
        self.alive.iter().filter(|&&a| a).count()
    }

    pub fn clear(&mut self) {
        self.vectors.clear();
        self.postings.clear();
        self.alive.clear();
    }
}

// -------------------------------------------------------------------------
// PyO3 wrapper
// -------------------------------------------------------------------------

#[pyclass]
pub struct RustSparseIndex {
    inner: RwLock<SparseIndex>,
    idx_to_id: RwLock<Vec<String>>,
    id_to_idx: RwLock<HashMap<String, usize>>,
}

#[pymethods]
impl RustSparseIndex {
    #[new]
    fn new() -> Self {
        Self {
            inner: RwLock::new(SparseIndex::new()),
            idx_to_id: RwLock::new(Vec::new()),
            id_to_idx: RwLock::new(HashMap::new()),
        }
    }

    /// Add a sparse vector. indices and values are parallel arrays.
    fn add_vector(
        &self,
        vector_id: String,
        indices: Vec<u32>,
        values: Vec<f32>,
    ) -> PyResult<()> {
        if indices.len() != values.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "indices and values must have same length"
            ));
        }
        if self.id_to_idx.read().contains_key(&vector_id) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Vector ID {} already exists", vector_id)
            ));
        }

        let vec = SparseVector::new(indices, values);
        let idx = self.inner.write().add(vec);

        self.id_to_idx.write().insert(vector_id.clone(), idx);
        let mut ids = self.idx_to_id.write();
        if ids.len() <= idx {
            ids.resize(idx + 1, String::new());
        }
        ids[idx] = vector_id;
        Ok(())
    }

    /// Search by sparse dot product.
    fn search<'py>(
        &self,
        py: Python<'py>,
        query_indices: Vec<u32>,
        query_values: Vec<f32>,
        k: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let query = SparseVector::new(query_indices, query_values);
        let results = self.inner.read().search(&query, k);

        let idx_to_id = self.idx_to_id.read();
        let result_list = PyList::empty(py);
        for (idx, score) in results {
            if idx < idx_to_id.len() && !idx_to_id[idx].is_empty() {
                result_list.append(
                    (idx_to_id[idx].as_str(), score)
                        .into_pyobject(py)
                        .unwrap()
                        .into_any()
                        .unbind(),
                )?;
            }
        }
        Ok(result_list)
    }

    fn remove_vector(&self, vector_id: String) -> bool {
        let idx = match self.id_to_idx.read().get(&vector_id) {
            Some(&idx) => idx,
            None => return false,
        };
        if self.inner.write().remove(idx) {
            self.id_to_idx.write().remove(&vector_id);
            true
        } else {
            false
        }
    }

    fn size(&self) -> usize {
        self.inner.read().len()
    }

    fn clear(&self) {
        self.inner.write().clear();
        self.idx_to_id.write().clear();
        self.id_to_idx.write().clear();
    }
}
