use pyo3::prelude::*;

mod distance;
mod storage;
mod fast_hnsw;
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
    // FastHNSW is the default — 12.5x faster than legacy
    m.add_class::<fast_hnsw::RustFastHNSWIndex>()?;

    // Legacy HNSW (archived, only compiled with --features legacy_hnsw)
    #[cfg(feature = "legacy_hnsw")]
    m.add_class::<legacy_hnsw::RustHNSWIndex>()?;

    m.add_class::<RustBruteForceIndex>()?;
    m.add_class::<RustLSHIndex>()?;
    m.add_class::<RustKDTreeIndex>()?;

    #[cfg(feature = "rabitq")]
    m.add_class::<rabitq_index::quantized::RustRabitqIndex>()?;

    Ok(())
}
