use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyReadonlyArray1, PyArray1, PyArray2};
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::sync::Arc;

/// Hash a vector with stable precision
fn hash_vector(vector: &[f32]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for &val in vector {
        // Round to 6 decimals for stable hashing
        let rounded = (val * 1_000_000.0).round() as i32;
        rounded.hash(&mut hasher);
    }
    hasher.finish()
}

/// Check if two vectors are approximately equal
fn vectors_equal(a: &[f32], b: &[f32], atol: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= atol)
}

struct VectorStoreInner {
    dimension: usize,
    vectors: Vec<Vec<f32>>,
    chunk_to_index: HashMap<String, usize>,
    ref_counts: HashMap<usize, usize>,
    vector_hashes: HashMap<u64, usize>,
    next_index: usize,
    free_indices: HashSet<usize>,
}

impl VectorStoreInner {
    fn new(dimension: usize, initial_capacity: usize) -> Self {
        Self {
            dimension,
            vectors: Vec::with_capacity(initial_capacity),
            chunk_to_index: HashMap::new(),
            ref_counts: HashMap::new(),
            vector_hashes: HashMap::new(),
            next_index: 0,
            free_indices: HashSet::new(),
        }
    }

    fn add_vector(&mut self, chunk_id: String, vector: Vec<f32>) -> PyResult<usize> {
        // Validate vector
        if vector.len() != self.dimension {
            return Err(PyValueError::new_err(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dimension,
                vector.len()
            )));
        }

        // Check if chunk already has a vector
        if self.chunk_to_index.contains_key(&chunk_id) {
            return Err(PyValueError::new_err(format!(
                "Chunk {} already has an associated vector",
                chunk_id
            )));
        }

        // Check for duplicate vector
        let vector_hash = hash_vector(&vector);
        if let Some(&index) = self.vector_hashes.get(&vector_hash) {
            // Verify it's actually identical (hash collision check)
            if vectors_equal(&self.vectors[index], &vector, 1e-6) {
                self.chunk_to_index.insert(chunk_id, index);
                *self.ref_counts.get_mut(&index).unwrap() += 1;
                return Ok(index);
            }
        }

        // Store new vector
        let index = self.allocate_index();
        self.ensure_capacity(index + 1);

        if index >= self.vectors.len() {
            self.vectors.resize(index + 1, Vec::new());
        }
        self.vectors[index] = vector;
        self.chunk_to_index.insert(chunk_id, index);
        self.ref_counts.insert(index, 1);
        self.vector_hashes.insert(vector_hash, index);

        Ok(index)
    }

    fn get_vector(&self, chunk_id: &str) -> Option<Vec<f32>> {
        self.chunk_to_index
            .get(chunk_id)
            .map(|&index| self.vectors[index].clone())
    }

    fn get_vector_by_index(&self, index: usize) -> PyResult<Vec<f32>> {
        if index >= self.next_index {
            return Err(PyValueError::new_err(format!(
                "Invalid vector index: {}",
                index
            )));
        }

        if self.free_indices.contains(&index) {
            return Err(PyValueError::new_err(format!(
                "Vector index {} has been freed",
                index
            )));
        }

        Ok(self.vectors[index].clone())
    }

    fn remove_vector(&mut self, chunk_id: &str) -> bool {
        if let Some(&index) = self.chunk_to_index.get(chunk_id) {
            // Remove chunk association
            self.chunk_to_index.remove(chunk_id);

            // Decrement reference count
            let ref_count = self.ref_counts.get_mut(&index).unwrap();
            *ref_count -= 1;

            // If no more references, free the vector
            if *ref_count == 0 {
                self.free_vector(index);
            }

            true
        } else {
            false
        }
    }

    fn get_vectors_by_indices(&self, indices: &[usize]) -> PyResult<Vec<Vec<f32>>> {
        let mut result = Vec::with_capacity(indices.len());
        for &idx in indices {
            if idx >= self.next_index {
                return Err(PyValueError::new_err(format!(
                    "Invalid vector index: {}",
                    idx
                )));
            }
            if self.free_indices.contains(&idx) {
                return Err(PyValueError::new_err(format!(
                    "Vector index {} has been freed",
                    idx
                )));
            }
            result.push(self.vectors[idx].clone());
        }
        Ok(result)
    }

    fn allocate_index(&mut self) -> usize {
        if let Some(&index) = self.free_indices.iter().next() {
            self.free_indices.remove(&index);
            index
        } else {
            let index = self.next_index;
            self.next_index += 1;
            index
        }
    }

    fn free_vector(&mut self, index: usize) {
        // Remove from reference count
        self.ref_counts.remove(&index);

        // Remove from hash map
        let vector_hash = hash_vector(&self.vectors[index]);
        if let Some(&hash_index) = self.vector_hashes.get(&vector_hash) {
            if hash_index == index {
                self.vector_hashes.remove(&vector_hash);
            }
        }

        // Zero out the vector
        self.vectors[index].clear();

        // Mark index as free
        self.free_indices.insert(index);
    }

    fn ensure_capacity(&mut self, required_size: usize) {
        if required_size <= self.vectors.capacity() {
            return;
        }

        // Grow by 50%
        let new_capacity = required_size.max((self.vectors.capacity() as f64 * 1.5) as usize);
        self.vectors.reserve(new_capacity - self.vectors.len());
    }

    fn count(&self) -> usize {
        self.next_index - self.free_indices.len()
    }

    fn total_references(&self) -> usize {
        self.ref_counts.values().sum()
    }
}

#[pyclass]
pub struct RustVectorStore {
    inner: Arc<RwLock<VectorStoreInner>>,
}

#[pymethods]
impl RustVectorStore {
    #[new]
    #[pyo3(signature = (dimension, initial_capacity=1000))]
    fn new(dimension: usize, initial_capacity: usize) -> PyResult<Self> {
        if dimension == 0 {
            return Err(PyValueError::new_err("Dimension must be positive"));
        }
        if initial_capacity == 0 {
            return Err(PyValueError::new_err(
                "Initial capacity must be positive",
            ));
        }

        Ok(Self {
            inner: Arc::new(RwLock::new(VectorStoreInner::new(
                dimension,
                initial_capacity,
            ))),
        })
    }

    fn add_vector<'py>(
        &self,
        chunk_id: String,
        vector: PyReadonlyArray1<f32>,
    ) -> PyResult<usize> {
        let vector_data = vector.as_slice()?.to_vec();
        self.inner.write().add_vector(chunk_id, vector_data)
    }

    fn get_vector<'py>(&self, py: Python<'py>, chunk_id: String) -> PyResult<Option<&'py PyArray1<f32>>> {
        let inner = self.inner.read();
        match inner.get_vector(&chunk_id) {
            Some(vector) => Ok(Some(PyArray1::from_vec(py, vector))),
            None => Ok(None),
        }
    }

    fn get_vector_by_index<'py>(
        &self,
        py: Python<'py>,
        index: usize,
    ) -> PyResult<&'py PyArray1<f32>> {
        let inner = self.inner.read();
        let vector = inner.get_vector_by_index(index)?;
        Ok(PyArray1::from_vec(py, vector))
    }

    fn remove_vector(&self, chunk_id: String) -> PyResult<bool> {
        Ok(self.inner.write().remove_vector(&chunk_id))
    }

    fn get_vectors_by_indices<'py>(
        &self,
        py: Python<'py>,
        indices: Vec<usize>,
    ) -> PyResult<&'py PyArray2<f32>> {
        let inner = self.inner.read();
        let vectors = inner.get_vectors_by_indices(&indices)?;

        if vectors.is_empty() {
            return Ok(PyArray2::zeros(py, (0, inner.dimension), false));
        }

        // Convert Vec<Vec<f32>> to Vec<&[f32]> for from_vec2
        let vec_refs: Vec<Vec<f32>> = vectors;
        Ok(PyArray2::from_vec2(py, &vec_refs)?)
    }

    fn size(&self) -> PyResult<usize> {
        Ok(self.inner.read().count())
    }

    fn total_references(&self) -> PyResult<usize> {
        Ok(self.inner.read().total_references())
    }

    fn dimension(&self) -> PyResult<usize> {
        Ok(self.inner.read().dimension)
    }

    fn get_statistics(&self) -> PyResult<HashMap<String, String>> {
        let inner = self.inner.read();
        let capacity = inner.vectors.capacity();
        let unique = inner.count();
        let utilization = if capacity > 0 {
            (unique as f64 / capacity as f64 * 100.0).round()
        } else {
            0.0
        };

        let mut stats = HashMap::new();
        stats.insert("unique_vectors".to_string(), unique.to_string());
        stats.insert(
            "total_references".to_string(),
            inner.total_references().to_string(),
        );
        stats.insert("capacity".to_string(), capacity.to_string());
        stats.insert("utilization".to_string(), format!("{:.2}", utilization));
        stats.insert(
            "free_indices".to_string(),
            inner.free_indices.len().to_string(),
        );
        stats.insert("dimension".to_string(), inner.dimension.to_string());
        stats.insert("storage_type".to_string(), "rust-in-memory".to_string());

        Ok(stats)
    }
}

#[pymodule]
fn rust_vector_store(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustVectorStore>()?;
    Ok(())
}
