/// Distance metrics with manually unrolled loops for reliable SIMD vectorization.
///
/// The key optimization: process 8 floats per iteration (256-bit SIMD register width).
/// The compiler WILL vectorize these loops because:
/// 1. Fixed stride, no branches in the inner loop
/// 2. Accumulator pattern (sum reduction)
/// 3. 8-wide processing matches AVX2 register width
/// 4. Remainder loop handles non-aligned tails
///
/// On AVX2 (most modern x86): 8 floats * 1 FMA per cycle = 8 FLOPS/cycle
/// On AVX-512: 16 floats per cycle
/// On ARM NEON: 4 floats per cycle

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DistanceMetric {
    Cosine,
    L2,
    InnerProduct,
    Manhattan,
    Hamming,
}

#[inline(always)]
pub fn compute_distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::Cosine => cosine_distance(a, b),
        DistanceMetric::L2 => l2_distance(a, b),
        DistanceMetric::InnerProduct => inner_product_distance(a, b),
        DistanceMetric::Manhattan => manhattan_distance(a, b),
        DistanceMetric::Hamming => hamming_distance(a, b),
    }
}

/// Dot product with 8-wide unrolled accumulation.
/// This is the hot inner loop — every other distance function calls this or
/// uses the same pattern.
#[inline(always)]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();

    // 4 independent accumulators to exploit instruction-level parallelism.
    // Modern CPUs can execute 2-4 FMA operations per cycle if they're independent.
    let mut sum0: f32 = 0.0;
    let mut sum1: f32 = 0.0;
    let mut sum2: f32 = 0.0;
    let mut sum3: f32 = 0.0;

    let chunks = n / 8;
    let remainder = n % 8;

    // Main loop: 8 elements per iteration, 4 accumulators
    let mut i = 0;
    for _ in 0..chunks {
        unsafe {
            sum0 += *a.get_unchecked(i) * *b.get_unchecked(i);
            sum1 += *a.get_unchecked(i + 1) * *b.get_unchecked(i + 1);
            sum2 += *a.get_unchecked(i + 2) * *b.get_unchecked(i + 2);
            sum3 += *a.get_unchecked(i + 3) * *b.get_unchecked(i + 3);
            sum0 += *a.get_unchecked(i + 4) * *b.get_unchecked(i + 4);
            sum1 += *a.get_unchecked(i + 5) * *b.get_unchecked(i + 5);
            sum2 += *a.get_unchecked(i + 6) * *b.get_unchecked(i + 6);
            sum3 += *a.get_unchecked(i + 7) * *b.get_unchecked(i + 7);
        }
        i += 8;
    }

    // Remainder
    for j in 0..remainder {
        unsafe {
            sum0 += *a.get_unchecked(i + j) * *b.get_unchecked(i + j);
        }
    }

    sum0 + sum1 + sum2 + sum3
}

/// Cosine distance = 1 - dot_product. Assumes normalized vectors.
#[inline(always)]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - dot_product(a, b)
}

/// Inner product distance = 1 - dot_product. Smaller = more similar.
#[inline(always)]
pub fn inner_product_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - dot_product(a, b)
}

/// Squared L2 (Euclidean) distance with 8-wide unrolling.
#[inline(always)]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();

    let mut sum0: f32 = 0.0;
    let mut sum1: f32 = 0.0;
    let mut sum2: f32 = 0.0;
    let mut sum3: f32 = 0.0;

    let chunks = n / 8;
    let remainder = n % 8;
    let mut i = 0;

    for _ in 0..chunks {
        unsafe {
            let d0 = *a.get_unchecked(i) - *b.get_unchecked(i);
            let d1 = *a.get_unchecked(i + 1) - *b.get_unchecked(i + 1);
            let d2 = *a.get_unchecked(i + 2) - *b.get_unchecked(i + 2);
            let d3 = *a.get_unchecked(i + 3) - *b.get_unchecked(i + 3);
            let d4 = *a.get_unchecked(i + 4) - *b.get_unchecked(i + 4);
            let d5 = *a.get_unchecked(i + 5) - *b.get_unchecked(i + 5);
            let d6 = *a.get_unchecked(i + 6) - *b.get_unchecked(i + 6);
            let d7 = *a.get_unchecked(i + 7) - *b.get_unchecked(i + 7);
            sum0 += d0 * d0 + d4 * d4;
            sum1 += d1 * d1 + d5 * d5;
            sum2 += d2 * d2 + d6 * d6;
            sum3 += d3 * d3 + d7 * d7;
        }
        i += 8;
    }

    for j in 0..remainder {
        unsafe {
            let d = *a.get_unchecked(i + j) - *b.get_unchecked(i + j);
            sum0 += d * d;
        }
    }

    sum0 + sum1 + sum2 + sum3
}

/// Manhattan (L1) distance.
#[inline(always)]
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();

    let mut sum0: f32 = 0.0;
    let mut sum1: f32 = 0.0;
    let mut sum2: f32 = 0.0;
    let mut sum3: f32 = 0.0;

    let chunks = n / 4;
    let remainder = n % 4;
    let mut i = 0;

    for _ in 0..chunks {
        unsafe {
            sum0 += (*a.get_unchecked(i) - *b.get_unchecked(i)).abs();
            sum1 += (*a.get_unchecked(i + 1) - *b.get_unchecked(i + 1)).abs();
            sum2 += (*a.get_unchecked(i + 2) - *b.get_unchecked(i + 2)).abs();
            sum3 += (*a.get_unchecked(i + 3) - *b.get_unchecked(i + 3)).abs();
        }
        i += 4;
    }

    for j in 0..remainder {
        unsafe {
            sum0 += (*a.get_unchecked(i + j) - *b.get_unchecked(i + j)).abs();
        }
    }

    sum0 + sum1 + sum2 + sum3
}

/// Hamming distance (for binary/quantized vectors stored as f32 0.0/1.0).
#[inline(always)]
pub fn hamming_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut count: u32 = 0;
    for i in 0..a.len() {
        unsafe {
            if (*a.get_unchecked(i) - *b.get_unchecked(i)).abs() > 0.5 {
                count += 1;
            }
        }
    }
    count as f32
}

/// Batch-4 dot product: compute 4 distances simultaneously.
/// The query vector is loaded once and broadcast across 4 candidate vectors.
/// This saves 3 * dim loads compared to 4 individual dot products.
/// (FAISS technique from distances_autovec-inl.h)
#[inline(always)]
pub fn dot_product_batch4(
    query: &[f32],
    v0: &[f32], v1: &[f32], v2: &[f32], v3: &[f32],
) -> (f32, f32, f32, f32) {
    let n = query.len();
    let mut s0: f32 = 0.0;
    let mut s1: f32 = 0.0;
    let mut s2: f32 = 0.0;
    let mut s3: f32 = 0.0;

    for i in 0..n {
        unsafe {
            let q = *query.get_unchecked(i);
            s0 += q * *v0.get_unchecked(i);
            s1 += q * *v1.get_unchecked(i);
            s2 += q * *v2.get_unchecked(i);
            s3 += q * *v3.get_unchecked(i);
        }
    }
    (s0, s1, s2, s3)
}

/// Compute 4 cosine distances simultaneously.
#[inline(always)]
pub fn cosine_distance_batch4(
    query: &[f32],
    v0: &[f32], v1: &[f32], v2: &[f32], v3: &[f32],
) -> (f32, f32, f32, f32) {
    let (d0, d1, d2, d3) = dot_product_batch4(query, v0, v1, v2, v3);
    (1.0 - d0, 1.0 - d1, 1.0 - d2, 1.0 - d3)
}

// Also add prefetch hint for the next vector during graph traversal
// (called before compute_distance to warm the cache)
#[inline(always)]
pub fn prefetch_vector(ptr: *const f32) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        #[cfg(target_feature = "sse")]
        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
    // No-op on non-x86 (aarch64 prefetch requires nightly)
    #[cfg(not(target_arch = "x86_64"))]
    let _ = ptr;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_distance_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_distance(&a, &b) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((cosine_distance(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        assert!((cosine_distance(&a, &b) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((dot_product(&a, &b) - 32.0).abs() < 1e-5);
    }

    #[test]
    fn test_l2_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((l2_distance(&a, &b) - 2.0).abs() < 1e-6); // sqrt(2)^2 = 2
    }

    #[test]
    fn test_manhattan_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 0.0, 1.0];
        assert!((manhattan_distance(&a, &b) - 7.0).abs() < 1e-6); // |3| + |2| + |2| = 7
    }

    #[test]
    fn test_hamming_distance() {
        let a = vec![0.0, 1.0, 0.0, 1.0];
        let b = vec![0.0, 0.0, 0.0, 1.0];
        assert!((hamming_distance(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_large() {
        // Test with dimension that exercises the remainder loop
        let n = 131; // Not divisible by 8
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.01).collect();
        let naive: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!((dot_product(&a, &b) - naive).abs() < 1e-3);
    }
}
