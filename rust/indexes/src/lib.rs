use pyo3::prelude::*;

mod distance;
mod storage;
mod fast_hnsw;
mod fast_search;
mod bm25;
mod sparse;
mod multi_vector;
mod brute_force;
mod lsh;
mod kd_tree;

#[cfg(feature = "legacy_hnsw")]
mod node;
#[cfg(feature = "legacy_hnsw")]
mod legacy_hnsw;

#[cfg(feature = "rabitq")]
mod rabitq_index;

pub use brute_force::RustBruteForceIndex;
pub use lsh::RustLSHIndex;
pub use kd_tree::RustKDTreeIndex;

/// Python module initialization.
#[pymodule]
fn rust_hnsw(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core indexes
    m.add_class::<fast_hnsw::RustFastHNSWIndex>()?;
    m.add_class::<bm25::RustBM25Index>()?;
    m.add_class::<sparse::RustSparseIndex>()?;
    m.add_class::<multi_vector::RustMultiVectorStore>()?;

    // Other index types
    m.add_class::<RustBruteForceIndex>()?;
    m.add_class::<RustLSHIndex>()?;
    m.add_class::<RustKDTreeIndex>()?;

    // Optional: legacy HNSW
    #[cfg(feature = "legacy_hnsw")]
    m.add_class::<legacy_hnsw::RustHNSWIndex>()?;

    // Optional: RaBitQ quantization
    #[cfg(feature = "rabitq")]
    m.add_class::<rabitq_index::quantized::RustRabitqIndex>()?;

    Ok(())
}
