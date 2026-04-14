/// RaBitQ quantized index via rabitq-rs.
///
/// Provides 4-32x memory compression with minimal recall loss.
/// Uses IVF+RaBitQ: vectors are clustered, then quantized within clusters.
///
/// Build pattern: buffer vectors via add_vector(), then call rebuild()
/// to construct the quantized index. Search is only available after rebuild.

#[cfg(feature = "rabitq")]
pub mod quantized {

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use numpy::PyReadonlyArray1;
use parking_lot::RwLock;
use std::collections::HashMap;

use rabitq_rs::{IvfRabitqIndex, Metric as RqMetric, RotatorType, SearchParams};

#[pyclass]
pub struct RustRabitqIndex {
    dim: usize,
    nlist: usize,
    total_bits: usize,
    nprobe: usize,
    metric: RqMetric,

    /// Buffered vectors (raw f32) before rebuild
    buffer_vectors: RwLock<Vec<Vec<f32>>>,
    /// String ID -> buffer index
    id_to_idx: RwLock<HashMap<String, usize>>,
    /// Buffer index -> String ID
    idx_to_id: RwLock<Vec<String>>,

    /// The quantized index (built on rebuild)
    index: RwLock<Option<IvfRabitqIndex>>,
}

#[pymethods]
impl RustRabitqIndex {
    #[new]
    #[pyo3(signature = (dimension, nlist=256, total_bits=7, nprobe=32, metric="l2"))]
    fn new(
        dimension: usize,
        nlist: usize,
        total_bits: usize,
        nprobe: usize,
        metric: &str,
    ) -> PyResult<Self> {
        let rq_metric = match metric {
            "l2" | "euclidean" => RqMetric::L2,
            "ip" | "inner_product" | "dot" | "cosine" => RqMetric::InnerProduct,
            _ => return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unknown metric. Use: l2, inner_product")
            )),
        };
        Ok(Self {
            dim: dimension,
            nlist,
            total_bits,
            nprobe,
            metric: rq_metric,
            buffer_vectors: RwLock::new(Vec::new()),
            id_to_idx: RwLock::new(HashMap::new()),
            idx_to_id: RwLock::new(Vec::new()),
            index: RwLock::new(None),
        })
    }

    fn add_vector(&self, vector_id: String, vector: PyReadonlyArray1<f32>) -> PyResult<()> {
        let data = vector.as_slice()?.to_vec();
        if data.len() != self.dim {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Vector dimension mismatch"
            ));
        }
        if self.id_to_idx.read().contains_key(&vector_id) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Vector ID {} already exists", vector_id)
            ));
        }

        let mut buf = self.buffer_vectors.write();
        let idx = buf.len();
        buf.push(data);

        self.id_to_idx.write().insert(vector_id.clone(), idx);
        self.idx_to_id.write().push(vector_id);

        Ok(())
    }

    /// Build the quantized index from buffered vectors.
    fn rebuild(&self) -> PyResult<()> {
        let buf = self.buffer_vectors.read();
        let n = buf.len();
        if n == 0 {
            *self.index.write() = None;
            return Ok(());
        }

        let actual_nlist = self.nlist.min(n / 10).max(1);

        let built = IvfRabitqIndex::train(
            &buf,
            actual_nlist,
            self.total_bits,
            self.metric,
            RotatorType::FhtKacRotator,
            42,
            false,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
            format!("RaBitQ build failed: {}", e)
        ))?;

        *self.index.write() = Some(built);
        Ok(())
    }

    #[pyo3(signature = (query_vector, k, distance_threshold=None, nprobe_override=None))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        query_vector: PyReadonlyArray1<f32>,
        k: usize,
        distance_threshold: Option<f32>,
        nprobe_override: Option<usize>,
    ) -> PyResult<&'py PyList> {
        let query = query_vector.as_slice()?.to_vec();
        let idx_guard = self.index.read();
        let index = match idx_guard.as_ref() {
            Some(idx) => idx,
            None => return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Index not built. Call rebuild() first."
            )),
        };

        let nprobe = nprobe_override.unwrap_or(self.nprobe);
        let params = SearchParams::new(k, nprobe);
        let results = index.search(&query, params)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Search failed: {}", e)
            ))?;

        let idx_to_id = self.idx_to_id.read();
        let result_list = PyList::empty(py);
        for r in results {
            if let Some(threshold) = distance_threshold {
                if r.score > threshold {
                    continue;
                }
            }
            if r.id < idx_to_id.len() {
                result_list.append(
                    (idx_to_id[r.id].as_str(), r.score).to_object(py)
                )?;
            }
        }
        Ok(result_list)
    }

    fn size(&self) -> usize {
        self.buffer_vectors.read().len()
    }

    fn clear(&self) {
        self.buffer_vectors.write().clear();
        self.id_to_idx.write().clear();
        self.idx_to_id.write().clear();
        *self.index.write() = None;
    }

    fn is_built(&self) -> bool {
        self.index.read().is_some()
    }

    fn get_statistics<'py>(&self, py: Python<'py>) -> PyResult<&'py PyDict> {
        let stats = PyDict::new(py);
        let n = self.buffer_vectors.read().len();
        stats.set_item("size", n)?;
        stats.set_item("dimension", self.dim)?;
        stats.set_item("nlist", self.nlist)?;
        stats.set_item("total_bits", self.total_bits)?;
        stats.set_item("nprobe", self.nprobe)?;
        stats.set_item("is_built", self.index.read().is_some())?;

        let raw_bytes = n * self.dim * 4;
        let quant_bytes = n * self.dim * self.total_bits / 8 + n * 40;
        stats.set_item("raw_memory_bytes", raw_bytes)?;
        stats.set_item("quantized_memory_bytes", quant_bytes)?;
        if raw_bytes > 0 {
            stats.set_item(
                "compression_ratio",
                raw_bytes as f64 / quant_bytes as f64,
            )?;
        }

        Ok(stats)
    }
}

} // mod quantized
