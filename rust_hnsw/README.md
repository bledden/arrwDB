# Rust HNSW Implementation for arrwDB

High-performance Hierarchical Navigable Small World (HNSW) index implementation in Rust with Python bindings via PyO3.

## Performance

Benchmarked on 10,000 vectors (384 dimensions):

| Metric | Python | Rust | Speedup |
|--------|--------|------|---------|
| **Search (avg)** | 1.64ms | 0.37ms | **4.45x** |
| **Build time** | 58.9s | 13.3s | **4.43x** |
| **Queries/sec** | 609 | 2,712 | **4.5x** |

## Features

- **SIMD-optimized distance calculations**: Auto-vectorization for cosine similarity
- **Thread-safe**: Uses `parking_lot::RwLock` for concurrent access
- **Zero-copy NumPy integration**: Direct memory access via PyO3
- **Memory efficient**: Optimized data structures with HashMap/HashSet
- **No GIL**: True parallelism for multi-threaded workloads

## Building

### Prerequisites

- Rust toolchain (install from https://rustup.rs/)
- Python 3.9+
- Maturin (`pip install maturin`)

### Build and Install

```bash
# From the rust_hnsw directory
python -m maturin build --release

# Install the wheel
pip install target/wheels/rust_hnsw-0.1.0-cp39-abi3-macosx_11_0_arm64.whl
```

### Development Build

For development with faster compilation:

```bash
python -m maturin develop
```

## Usage

### Direct Usage

```python
import rust_hnsw
import numpy as np

# Create index
index = rust_hnsw.RustHNSWIndex(
    dimension=384,
    m=16,                    # Max connections per layer
    ef_construction=200,     # Quality during build
    ef_search=50            # Quality during search
)

# Add vectors
vector = np.random.randn(384).astype(np.float32)
vector = vector / np.linalg.norm(vector)  # Normalize
index.add_vector("vec-001", vector)

# Search
query = np.random.randn(384).astype(np.float32)
query = query / np.linalg.norm(query)
results = index.search(query, k=10)

# Results are list of (vector_id, distance) tuples
for vid, distance in results:
    print(f"ID: {vid}, Distance: {distance:.4f}")

# Get statistics
stats = index.get_statistics()
print(f"Index size: {stats['size']}")
print(f"Levels: {stats.get('num_levels', 'N/A')}")
```

### Integration with arrwDB

Use the wrapper for seamless integration:

```python
from infrastructure.indexes.rust_hnsw_wrapper import RustHNSWIndexWrapper
from core.vector_store import VectorStore

# Create vector store
vector_store = VectorStore(dimension=384)

# Create Rust HNSW index (drop-in replacement for Python version)
index = RustHNSWIndexWrapper(
    vector_store=vector_store,
    M=16,
    ef_construction=200,
    ef_search=50
)

# Use exactly like the Python HNSWIndex
from uuid import uuid4

vector_id = uuid4()
vector = np.random.randn(384).astype(np.float32)
vector = vector / np.linalg.norm(vector)

idx = vector_store.add_vector(vector_id, vector)
index.add_vector(vector_id, idx)

# Search
results = index.search(query_vector, k=10)
```

## API Reference

### RustHNSWIndex

```python
class RustHNSWIndex:
    def __init__(
        self,
        dimension: int,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        max_level: int = 16
    )
```

**Parameters:**
- `dimension`: Dimensionality of vectors
- `m`: Maximum connections per node (higher = better recall, more memory)
- `ef_construction`: Construction quality (higher = better index, slower build)
- `ef_search`: Search quality (higher = better recall, slower search)
- `max_level`: Maximum graph hierarchy depth

**Methods:**
- `add_vector(vector_id: str, vector: np.ndarray)`: Add a vector
- `remove_vector(vector_id: str) -> bool`: Remove a vector
- `search(query: np.ndarray, k: int, distance_threshold: float = None) -> List[Tuple[str, float]]`: Search for k-NN
- `size() -> int`: Get number of vectors
- `clear()`: Remove all vectors
- `rebuild()`: Rebuild index from scratch
- `get_statistics() -> dict`: Get index statistics

## Benchmarking

Run the included benchmark script:

```bash
python benchmark.py
```

This will compare Python vs Rust performance on:
- Index building (10,000 vectors)
- Search queries (1,000 queries)
- Various k values

## Architecture

### Modules

- **lib.rs**: Main PyO3 bindings and HNSW algorithm
- **node.rs**: HNSW node data structure
- **distance.rs**: SIMD-optimized distance calculations

### Key Optimizations

1. **SIMD Distance Calculations**: Iterator-based dot product enables auto-vectorization
2. **Lock-Free Reads**: `RwLock` allows concurrent read access
3. **Zero-Copy Arrays**: `PyReadonlyArray1` provides direct memory access
4. **Efficient Collections**: `HashMap` and `HashSet` for O(1) lookups
5. **Release Optimizations**: LTO and single codegen unit for maximum performance

## Testing

Run Rust tests:

```bash
cargo test
```

Run integration tests with Python:

```bash
pytest tests/integration/test_rust_hnsw.py
```

## Troubleshooting

### Build Errors

**Python symbols not found:**
- Use `maturin` instead of `cargo build`
- Ensure Python development headers are installed

**Wrong architecture:**
- Build targets your system architecture automatically
- For cross-compilation, use `maturin build --target <triple>`

### Runtime Errors

**Import error:**
```python
ImportError: No module named 'rust_hnsw'
```
- Ensure wheel is installed: `pip install target/wheels/*.whl`

**Dimension mismatch:**
```python
ValueError: Vector dimension X doesn't match index dimension Y
```
- Verify all vectors have the same dimension as the index

## Performance Tips

1. **Tune ef_construction**: Higher values (200-500) improve index quality
2. **Tune ef_search**: Adjust based on recall/speed tradeoff
3. **Normalize vectors**: HNSW expects normalized vectors for cosine distance
4. **Batch operations**: Add vectors in batches when possible
5. **Use rebuild()**: After many deletions, rebuild for optimal performance

## Future Enhancements

- [ ] Parallel index building with Rayon
- [ ] Persistence (save/load index)
- [ ] Distance metrics (L2, inner product)
- [ ] Advanced pruning heuristics
- [ ] Memory-mapped storage for large indices
- [ ] GPU acceleration for distance calculations

## License

Same as arrwDB project.

## Contributing

Improvements welcome! Focus areas:
- Additional distance metrics
- Persistence layer
- Advanced optimizations
- Better error handling
