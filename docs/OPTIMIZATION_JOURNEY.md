# arrwDB Optimization Journey

## From 52 QPS to 1,834 QPS — 35x improvement on SIFT-1M

This document records every optimization applied to arrwDB's HNSW search engine, what worked, what didn't, and why.

### Hardware

All CPU benchmarks on GCP n2-highmem-16 (16 vCPU, 128GB RAM, AMD EPYC).
GPU benchmarks on GCP g2-standard-8 + NVIDIA L4 (24GB VRAM).
Dataset: SIFT-1M (1M vectors, 128 dimensions), M=48, ef_construction=400.

---

## What Worked

### 1. Integer-indexed storage (12.5x)
**52 → 648 QPS**

Replaced `HashMap<String, Vec<f32>>` for vectors and `HashMap<String, HNSWNode>` for graph nodes with contiguous `Vec<f32>` arrays and `Vec<Vec<usize>>` neighbor lists. String→usize mapping happens only at the PyO3 boundary.

Before: every neighbor visit required String clone + HashMap hash + RwLock acquire.
After: every neighbor visit is an array index.

### 2. SIMD-friendly unrolled distance (1.9x)
**648 → 1,239 QPS**

Replaced naive iterator `.zip().map().sum()` with 8-wide unrolled loops using 4 independent accumulators. This helps the compiler auto-vectorize and exploits instruction-level parallelism.

### 3. Explicit AVX2/FMA intrinsics + function pointer dispatch (1.5x)
**1,239 → 1,834 QPS**

Replaced auto-vectorized scalar code with explicit `_mm256_fmadd_ps` (fused multiply-add) intrinsics. 4 independent 256-bit accumulators process 32 floats per iteration. Runtime detection via `is_x86_feature_detected!` with scalar fallback.

Also: replaced `match metric` dispatch per distance call with a function pointer (`DistanceFn`) set once at construction time. Eliminated `unsafe get_unchecked` bounds checks on neighbor and alive arrays.

### 4. Visited list pool with generation counter (included in above)

Replaced `vec![false; 1_000_000]` allocation per search query with a thread-local `VisitedList` using a u16 generation counter. O(1) reset by incrementing the counter. Full memset only every 65,535 searches. Zero heap allocation in the search hot path.

### 5. target-cpu=native (included in above)

Added `.cargo/config.toml` with `rustflags = ["-C", "target-cpu=native"]` for x86_64 and aarch64 targets. Tells the compiler the exact CPU features available, enabling FMA and AVX2 instruction selection without runtime checks.

### 6. Recall quality fix

Changed construction search ef from `ef_construction` to `max(ef_construction, M_max)` matching hnswlib behavior. Ensures enough candidates are explored during graph construction.

---

## What Didn't Work

### Co-located memory layout (-10%)
**1,834 → 1,655 QPS (regression)**

Attempted to store neighbor lists and vector data adjacent in a single byte array (the hnswlib technique). At 128 dimensions with M_max0=96, each node's co-located block is 898 bytes (14 cache lines). Prefetching 2 cache lines only warms the neighbor list + 20% of the vector. The separate contiguous VectorStorage benefits from the hardware sequential prefetcher which detects the stride pattern for vector reads.

**Conclusion:** Co-located layout helps at dim ≤ 32 (entire node fits in 1-2 cache lines) but hurts at typical embedding dimensions (128-1024).

### Visited array L1 prefetch (-10%)
**1,834 → 1,637 QPS (regression)**

Attempted to prefetch the visited array entry for the next neighbor alongside the vector prefetch (3-prong prefetch from hnswlib). The visited array for 1M entries is 2MB (u16), which fits in L2 but not L1. Prefetching to L1 (T0 hint) evicted useful vector data from L1 cache.

**Conclusion:** The CPU handles visited array access fine via normal load → L2 hit. Forcing it to L1 causes cache pollution.

### Cached current distance in greedy navigation (noise)

Cached `dist(query, current)` to avoid recomputing for each neighbor comparison in upper-layer greedy navigation. Theoretically saves one dot product per neighbor. In practice, the effect was within cloud VM noise (~12% variance).

### Batch-4 distance computation (not measured in isolation)

Implemented FAISS's technique of computing 4 dot products simultaneously by loading the query once and broadcasting across 4 candidates. Included in the co-located search path which regressed, so the isolated effect is unknown.

---

## Build Rate Progression

| Version | Rate | Time (1M) |
|---------|------|-----------|
| Old HNSW (String/HashMap) | 26/s | 10.6 hours |
| FastHNSW v1 | 106/s | 2.6 hours |
| + SIMD | 256/s | 65 min |
| + AVX2/FMA + build_bulk | 346/s | 48 min |

---

## Competitive Position

| System | QPS @0.99 recall | Language | Our gap |
|--------|-----------------|----------|---------|
| hnswlib | 2,755 | C++ | 1.5x |
| ScaNN | 2,743 | C++ | 1.5x |
| **arrwDB** | **1,834** | **Rust** | — |
| FAISS-HNSW | 1,787 | C++ | **We win** |
| Weaviate | 913 | Go | **2x faster** |
| Qdrant | 572 | Rust | **3.2x faster** |

Note: ann-benchmarks numbers are on r6i.16xlarge (Intel Xeon). Our numbers are on n2-highmem-16 (AMD EPYC). Not directly comparable. AWS benchmark pending.

---

## What the Remaining 1.5x Gap Is

1. **hnswlib's co-located layout works because their stride is compile-time constant** — templates specialize per dimension. Our stride is a runtime field.
2. **hnswlib's distance function is a function pointer selected at construction** — we do the same now, but their entire search function is templated on dimension.
3. **hnswlib has zero abstraction overhead** — no PyO3, no RwLock, no VectorStore enum match. Pure C++ with raw pointers.
4. **10+ years of micro-tuning** — prefetch depths, branch hints, memory alignment.

To close this gap, arrwDB would need compile-time dimension specialization (Rust generics with const generics) and elimination of all remaining abstraction in the search path.

---

## GPU Results (separate from CPU optimization)

NVIDIA L4 via FAISS-GPU CAGRA:
- SIFT-1M: 3,175 QPS at 0.992 recall, **build in 27 seconds**
- Deep-1M: 2,763 QPS at 1.000 recall, build in 22 seconds
- Batch throughput: 78,799 - 105,891 QPS

GPU CAGRA is available as an alternative backend behind the same VectorIndex interface.
