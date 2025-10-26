# Quantization Design: User-Controlled Precision Trade-offs

**Philosophy**: Let users choose their own precision/memory trade-off based on their needs.

---

## Design Principles

1. **Opt-In by Default**: Float32 is default (no surprises)
2. **Per-Library Configuration**: Different libraries can use different strategies
3. **Transparent Trade-offs**: Show memory savings and accuracy impact
4. **Hybrid Approach**: Best of both worlds available
5. **Easy Migration**: Can change quantization strategy later

---

## Quantization Strategies

### **Strategy 1: None (Default)**
```python
library = create_library(
    name="high_precision_lib",
    quantization_config={
        "strategy": "none"  # Full float32
    }
)
```

**When to use**:
- Default choice (safest)
- Low-dimensional vectors (< 64 dims)
- Accuracy-critical applications
- Unlimited RAM budget

**Stats**:
- Memory: 100% (baseline)
- Accuracy: 100%
- Speed: 1x

---

### **Strategy 2: Scalar Quantization (int8)**
```python
library = create_library(
    name="efficient_lib",
    quantization_config={
        "strategy": "scalar",
        "bits": 8,  # 8-bit per dimension
        "calibration_samples": 10000  # Use first 10K vectors to learn min/max
    }
)
```

**When to use**:
- Most text embeddings (384+ dims)
- Memory-constrained deployments
- Cost optimization priority

**Stats**:
- Memory: 25% (4x compression)
- Accuracy: ~98-99% (1-2% loss)
- Speed: 2x faster (better cache)

**How it works**:
```python
# Learn per-dimension min/max from sample data
dimension_0: min=-0.5, max=0.8 → scale to 0-255
dimension_1: min=-0.3, max=1.2 → scale to 0-255
...

# Each value: float32 (4 bytes) → uint8 (1 byte)
```

---

### **Strategy 3: Product Quantization (PQ)**
```python
library = create_library(
    name="compressed_lib",
    quantization_config={
        "strategy": "product",
        "subvectors": 48,  # Split 384 dims into 48 groups of 8
        "codebook_size": 256,  # 256 codes per subvector
        "training_samples": 50000  # Learn from 50K vectors
    }
)
```

**When to use**:
- Very large datasets (10M+ vectors)
- Need 8-16x compression
- Can tolerate 2-3% accuracy loss

**Stats**:
- Memory: 12.5% (8x compression)
- Accuracy: ~97-98% (2-3% loss)
- Speed: 3x faster

**How it works**:
```python
# Learn optimal "codebooks" from training data
# Each subvector approximated by nearest code
# Store code indices (1 byte each) instead of floats
```

---

### **Strategy 4: Hybrid (Best of Both Worlds)** ⭐
```python
library = create_library(
    name="smart_lib",
    quantization_config={
        "strategy": "hybrid",
        "initial_search": "scalar",  # Use quantized for speed
        "rerank_top_k": 100,         # Rerank top 100 with float32
        "rerank_precision": "float32"
    }
)
```

**When to use**:
- Want speed AND accuracy
- Willing to use 125% memory (quantized + small float32 cache)
- Best user experience

**Stats**:
- Memory: 125% (quantized full + float32 cache for top results)
- Accuracy: ~99.5% (0.5% loss)
- Speed: 2x faster (quantized initial search)

**How it works**:
```python
# Step 1: Fast quantized search → get top 1000 candidates
candidates = quantized_search(query, k=1000)  # Fast!

# Step 2: Rerank top 100 with float32 → precise ranking
final_results = float32_rerank(candidates[:100])  # Accurate!

# Return top K
return final_results[:k]
```

**Why it's smart**:
- Initial search doesn't need perfect precision (just find candidates)
- Final ranking needs precision (users see these results)
- 90% of work done with quantized (fast)
- 10% of work done with float32 (accurate)

---

### **Strategy 5: Adaptive (Automatic)** 🤖
```python
library = create_library(
    name="adaptive_lib",
    quantization_config={
        "strategy": "adaptive",
        "max_memory_gb": 16,  # Constraint: stay under 16GB
        "target_accuracy": 0.98,  # Target: 98% recall@10
        "auto_tune": True
    }
)
```

**When to use**:
- Not sure what to choose
- Want system to optimize automatically
- Have clear constraints (memory or accuracy)

**How it works**:
```python
# System monitors your data and workload
# Automatically chooses best strategy

if vector_count < 100K and memory_available > 4GB:
    use_strategy = "none"  # No need to quantize
elif dimension < 64:
    use_strategy = "none"  # Low-dim = keep precision
elif memory_pressure > 0.8:
    use_strategy = "product"  # Aggressive compression
else:
    use_strategy = "scalar"  # Balanced
```

---

## API Design

### **Create Library with Quantization**
```python
POST /v1/libraries
{
    "name": "my_documents",
    "index_type": "hnsw",
    "quantization_config": {
        "strategy": "hybrid",
        "initial_search": "scalar",
        "rerank_top_k": 100
    }
}

# Response includes quantization info
{
    "id": "lib_123",
    "name": "my_documents",
    "quantization": {
        "strategy": "hybrid",
        "memory_savings": "75%",
        "estimated_accuracy": "99.5%"
    }
}
```

### **Get Library Statistics (Shows Quantization Impact)**
```python
GET /v1/libraries/{id}/statistics

{
    "library_id": "lib_123",
    "num_vectors": 1000000,
    "memory_usage": {
        "quantized": "384 MB",
        "float32_cache": "96 MB",
        "total": "480 MB",
        "savings_vs_float32": "75%"  # Would be 1920 MB
    },
    "quantization": {
        "strategy": "hybrid",
        "accuracy_metrics": {
            "recall@10": 0.995,
            "recall@100": 0.998
        }
    }
}
```

### **Change Quantization Strategy (Migration)**
```python
POST /v1/libraries/{id}/quantization/migrate
{
    "new_strategy": "scalar",
    "preserve_data": true  # Keep original float32 during migration
}

# Response
{
    "migration_id": "mig_456",
    "status": "in_progress",
    "eta_seconds": 120
}
```

---

## Implementation Phases

### **Phase 1: Scalar Quantization** (Week 1)
- [ ] Implement int8 scalar quantization in Rust
- [ ] Per-dimension min/max calibration
- [ ] Add `quantization_config` to library creation
- [ ] Update vector store to support quantized storage
- [ ] Modify HNSW index to work with quantized vectors

**Deliverable**: Users can opt-in to 4x memory savings

---

### **Phase 2: Hybrid Mode** (Week 2)
- [ ] Implement two-stage search (quantized → float32)
- [ ] Add float32 cache for top candidates
- [ ] Automatic cache management (LRU eviction)
- [ ] Benchmark accuracy vs float32 baseline

**Deliverable**: Best of both worlds (speed + accuracy)

---

### **Phase 3: Product Quantization** (Week 3)
- [ ] Implement PQ training (k-means clustering)
- [ ] Codebook storage and lookup
- [ ] Asymmetric distance computation (query float32, db PQ)
- [ ] Migration tool (none → scalar → PQ)

**Deliverable**: 8-16x compression for massive datasets

---

### **Phase 4: Adaptive Selection** (Week 4)
- [ ] Auto-tune based on data characteristics
- [ ] Memory pressure monitoring
- [ ] Accuracy benchmarking on sample data
- [ ] Strategy recommendation API

**Deliverable**: System auto-optimizes for user's workload

---

## Benchmarking & Transparency

### **Show Users What They're Getting**
```python
POST /v1/libraries/{id}/quantization/benchmark
{
    "strategies": ["none", "scalar", "hybrid", "product"],
    "test_queries": 100
}

# Response shows trade-offs
{
    "results": [
        {
            "strategy": "none",
            "memory_mb": 1920,
            "recall@10": 1.000,
            "avg_latency_ms": 50
        },
        {
            "strategy": "scalar",
            "memory_mb": 480,
            "recall@10": 0.989,  # ← 1.1% loss
            "avg_latency_ms": 25  # ← 2x faster
        },
        {
            "strategy": "hybrid",
            "memory_mb": 600,
            "recall@10": 0.997,  # ← 0.3% loss
            "avg_latency_ms": 30  # ← 1.7x faster
        }
    ],
    "recommendation": "hybrid"  # Best balance
}
```

**Let users make informed decisions!**

---

## Configuration Examples

### **Example 1: High-Accuracy Research**
```python
{
    "quantization_config": {
        "strategy": "none"
    }
}
# Use case: Scientific research, need exact values
```

### **Example 2: Cost-Optimized SaaS**
```python
{
    "quantization_config": {
        "strategy": "scalar",
        "bits": 8
    }
}
# Use case: Reduce cloud costs, 1-2% loss acceptable
```

### **Example 3: Large-Scale Search Engine**
```python
{
    "quantization_config": {
        "strategy": "hybrid",
        "initial_search": "product",  # 8x compression
        "rerank_top_k": 100,
        "rerank_precision": "float32"
    }
}
# Use case: Billion-scale, need speed + accuracy
```

### **Example 4: Unknown Workload**
```python
{
    "quantization_config": {
        "strategy": "adaptive",
        "max_memory_gb": 32,
        "target_accuracy": 0.98
    }
}
# Use case: Not sure, let system decide
```

---

## Migration Path

### **Start Safe, Optimize Later**
```python
# Day 1: Launch with float32 (safe)
library = create_library(quantization="none")

# Week 2: Users love it, but RAM costs high
# Run benchmark to see options
benchmark = run_quantization_benchmark(library)
# Shows: scalar = 4x memory savings, 1% accuracy loss

# Week 3: Migrate to scalar
migrate_quantization(library, strategy="scalar")
# Cloud bill drops by 75%!

# Month 2: Dataset grows to 10M vectors
# Migrate to hybrid for best balance
migrate_quantization(library, strategy="hybrid")
```

---

## Key Insight

**You're absolutely right**: Quantization is a trade-off, not a win-win.

**The solution**: Give users **control** and **visibility**
- Show memory savings
- Show accuracy impact
- Let them choose
- Make it easy to change later

**Best approach**:
1. Start with float32 (safe default)
2. Offer quantization as upgrade (cost savings)
3. Provide benchmarking tools (informed decisions)
4. Support hybrid mode (best of both worlds)

**Users appreciate transparency more than false promises!**

---

## Next Steps

Want me to:
1. ✅ Design the API endpoints for quantization config?
2. ✅ Implement scalar quantization (int8) first?
3. ✅ Build the benchmarking tool?
4. ✅ Create migration utilities?

Or keep this as design doc and move on to other features?
