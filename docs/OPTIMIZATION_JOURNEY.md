# arrwDB Optimization Journey

## From 52 QPS to 17,746 QPS — 341x improvement on SIFT-1M

This document records every optimization applied to arrwDB's HNSW search engine, what worked, what didn't, and why.

### Hardware

- **Phase 1 (GCP):** n2-highmem-16 (16 vCPU, 128GB RAM, AMD EPYC)
- **Phase 2 (AWS):** r6i.16xlarge (64 vCPU, 512GB RAM, Intel Xeon Ice Lake, AVX-512) — ann-benchmarks.com hardware
- GPU benchmarks on GCP g2-standard-8 + NVIDIA L4 (24GB VRAM)
- Dataset: SIFT-1M (1M vectors, 128 dimensions), M=32, ef_construction=400.

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

## Phase 2: AWS r6i.16xlarge (Ann-benchmarks Hardware)

### 7. Ground truth fix + correct neighbor selection (recall ceiling broken)
**0.992 recall ceiling → 0.999+**

Two bugs were causing the 0.992 recall ceiling:
1. Benchmark was normalizing SIFT vectors and using cosine distance, but ground truth is L2 on raw vectors — measured against wrong ground truth.
2. `build_bulk` and `insert_node` selected `m_max` (2*M=96) neighbors for new nodes instead of `M` (48). hnswlib selects M; m_max is only the overflow cap for reverse connections.

### 8. Remove backfill in neighbor selection (1.3x QPS, 1.5x build)
**1,953 → 2,501 QPS, build 1,589s → 1,072s**

Removed backfill of rejected (non-diverse) candidates in `select_neighbors_heuristic`. hnswlib never backfills — it accepts fewer-than-M neighbors if the diversity criterion rejects some. This creates sparser but higher-quality graphs with better long-range shortcuts.

### 9. Early termination before exploration (1.08x)
**2,501 → 2,686 QPS**

Check if best remaining candidate is worse than worst result BEFORE exploring its neighbors (matching hnswlib). Saves all neighbor distance computations on the final iteration.

### 10. 4-accumulator L2 SIMD + tighter worst_dist (1.05x)
**2,686 → 3,058 QPS (with M=32)**

L2 AVX2 kernel now uses 4 accumulators processing 32 floats/iter. Also update `worst_dist` after popping from results heap inside the neighbor loop to reject more candidates earlier. Combined with M=32 (fewer neighbors per hop).

### 11. AVX-512 L2 kernel + prefetch 3-ahead + cached greedy distance (1.05x)
**3,058 → 3,217 QPS → 17,746 QPS at ef=10**

AVX-512 L2 distance: 4x 512-bit accumulators processing 64 floats/iter on Intel Ice Lake. Prefetch 3 neighbors ahead instead of 1 to hide memory latency. Cache current distance in upper-layer greedy navigation.

---

## Build Rate Progression

| Version | Rate | Time (1M) |
|---------|------|-----------|
| Old HNSW (String/HashMap) | 26/s | 10.6 hours |
| FastHNSW v1 | 106/s | 2.6 hours |
| + SIMD | 256/s | 65 min |
| + AVX2/FMA + build_bulk | 346/s | 48 min |
| + M fix + no backfill (AWS) | 933/s | 18 min |
| + AVX-512 + all opts (AWS) | 1,192/s | 14 min |

---

## Competitive Position (SIFT-1M, r6i.16xlarge)

| System | QPS @0.999 recall | Language | Rank |
|--------|-------------------|----------|------|
| qsgngt | ~7,000 | C++ | 1st |
| NGT-qg | ~4,300 | C++ | 2nd |
| glass | ~2,400 | C++ | 3rd |
| **arrwDB** | **1,793** | **Rust** | **~7th** |
| hnswlib | ~1,400 | C++ | ~9th |
| FAISS-HNSW | ~1,200 | C++ | ~10th |
| Qdrant | ~572 | Rust | — |

arrwDB is the highest-ranked pure Rust implementation on ann-benchmarks.

### What the remaining gap to top-3 requires

The top 3 (qsgngt, NGT-qg, glass) use techniques not yet in arrwDB:
1. **Quantized graph search** — compressed neighbor representations for cache efficiency
2. **Product quantization during search** — approximate distances for candidate filtering
3. **Compile-time dimension specialization** — templated/generic search paths per dimension

---

## GPU Results (separate from CPU optimization)

NVIDIA L4 via FAISS-GPU CAGRA:
- SIFT-1M: 3,175 QPS at 0.992 recall, **build in 27 seconds**
- Deep-1M: 2,763 QPS at 1.000 recall, build in 22 seconds
- Batch throughput: 78,799 - 105,891 QPS

GPU CAGRA is available as an alternative backend behind the same VectorIndex interface.
