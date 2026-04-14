/// Multi-vector support: store multiple named vectors per record.
///
/// Each record can have vectors for different fields (e.g., "title", "content",
/// "image") with independent dimensions and distance metrics.
///
/// Usage:
///   store = RustMultiVectorStore()
///   store.add_field("title", dimension=384, metric="cosine")
///   store.add_field("content", dimension=1024, metric="cosine")
///   store.set_vector("doc1", "title", title_embedding)
///   store.set_vector("doc1", "content", content_embedding)
///   results = store.search("title", query_embedding, k=10)

use std::collections::HashMap;
use parking_lot::RwLock;
use pyo3::prelude::*;
use pyo3::types::PyList;
use numpy::PyReadonlyArray1;

use crate::fast_hnsw::FastHNSW;
use crate::distance::DistanceMetric;

/// Configuration for a named vector field.
struct FieldConfig {
    dimension: usize,
    metric: DistanceMetric,
}

#[pyclass]
pub struct RustMultiVectorStore {
    /// field_name -> HNSW index for that field
    fields: RwLock<HashMap<String, FastHNSW>>,
    /// field_name -> config
    configs: RwLock<HashMap<String, FieldConfig>>,
    /// record_id -> { field_name -> internal vector index }
    records: RwLock<HashMap<String, HashMap<String, usize>>>,
}

#[pymethods]
impl RustMultiVectorStore {
    #[new]
    fn new() -> Self {
        Self {
            fields: RwLock::new(HashMap::new()),
            configs: RwLock::new(HashMap::new()),
            records: RwLock::new(HashMap::new()),
        }
    }

    /// Register a named vector field with its dimension and metric.
    #[pyo3(signature = (field_name, dimension, m=16, ef_construction=200, ef_search=50, metric="cosine"))]
    fn add_field(
        &self,
        field_name: String,
        dimension: usize,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        metric: &str,
    ) -> PyResult<()> {
        let dist_metric = match metric {
            "cosine" => DistanceMetric::Cosine,
            "l2" | "euclidean" => DistanceMetric::L2,
            "ip" | "inner_product" => DistanceMetric::InnerProduct,
            _ => return Err(pyo3::exceptions::PyValueError::new_err("Unknown metric")),
        };

        if self.fields.read().contains_key(&field_name) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Field '{}' already exists", field_name)
            ));
        }

        let index = FastHNSW::new(dimension, m, ef_construction, ef_search, 16, dist_metric);
        self.fields.write().insert(field_name.clone(), index);
        self.configs.write().insert(field_name, FieldConfig { dimension, metric: dist_metric });
        Ok(())
    }

    /// Set a vector for a record in a specific field.
    /// If the record already has a vector in this field, it is overwritten.
    fn set_vector(
        &self,
        record_id: String,
        field_name: String,
        vector: PyReadonlyArray1<f32>,
    ) -> PyResult<()> {
        let data = vector.as_slice()?;

        let fields = self.fields.read();
        let index = fields.get(&field_name).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("Field '{}' not registered", field_name))
        })?;

        // Check if this record already has a vector in this field
        let existing_idx = {
            let records = self.records.read();
            records.get(&record_id)
                .and_then(|fields| fields.get(&field_name))
                .copied()
        };

        let idx = match existing_idx {
            Some(idx) => {
                // Upsert: update in place
                let (new_idx, _) = index.upsert_vector(Some(idx), data);
                new_idx
            }
            None => {
                // New insert
                index.add_vector(data)
            }
        };

        // Record the mapping
        self.records.write()
            .entry(record_id)
            .or_insert_with(HashMap::new)
            .insert(field_name, idx);

        Ok(())
    }

    /// Search a specific field for nearest neighbors.
    fn search<'py>(
        &self,
        py: Python<'py>,
        field_name: String,
        query_vector: PyReadonlyArray1<f32>,
        k: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let query = query_vector.as_slice()?;

        let fields = self.fields.read();
        let index = fields.get(&field_name).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("Field '{}' not registered", field_name))
        })?;

        let results = index.search(query, k, index.ef_search);

        // Reverse-map internal indices to record IDs
        let records = self.records.read();
        let result_list = PyList::empty(py);

        // Build reverse index: (field, internal_idx) -> record_id
        // This is O(n) but only needed at search time for k results
        let mut idx_to_record: HashMap<usize, &str> = HashMap::new();
        for (record_id, field_map) in records.iter() {
            if let Some(&vidx) = field_map.get(&field_name) {
                idx_to_record.insert(vidx, record_id.as_str());
            }
        }

        for (idx, dist) in results {
            if let Some(&record_id) = idx_to_record.get(&idx) {
                result_list.append(
                    (record_id, dist)
                        .into_pyobject(py)
                        .unwrap()
                        .into_any()
                        .unbind(),
                )?;
            }
        }
        Ok(result_list)
    }

    /// Get the list of registered field names.
    fn list_fields(&self) -> Vec<String> {
        self.fields.read().keys().cloned().collect()
    }

    fn size(&self) -> usize {
        self.records.read().len()
    }

    fn clear(&self) {
        self.fields.write().clear();
        self.configs.write().clear();
        self.records.write().clear();
    }
}
