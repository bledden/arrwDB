/// Distance metrics with explicit SIMD intrinsics.
///
/// x86_64 path: AVX2 + FMA intrinsics for guaranteed vectorization.
/// - _mm256_fmadd_ps: fused multiply-add (1 instruction for a*b+c)
/// - _mm256_loadu_ps: unaligned 256-bit load (8 floats)
/// - Processing 32 floats per iteration (4 x 8-wide FMA)
///
/// Fallback path: unrolled scalar (auto-vectorized) for ARM/other.

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

// ==========================================================================
// AVX2 + FMA intrinsics (x86_64)
// ==========================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_product_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    // 4 independent 256-bit accumulators (32 floats in flight)
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    let chunks = n / 32;
    let mut i = 0usize;

    for _ in 0..chunks {
        let a0 = _mm256_loadu_ps(a_ptr.add(i));
        let b0 = _mm256_loadu_ps(b_ptr.add(i));
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);

        let a1 = _mm256_loadu_ps(a_ptr.add(i + 8));
        let b1 = _mm256_loadu_ps(b_ptr.add(i + 8));
        acc1 = _mm256_fmadd_ps(a1, b1, acc1);

        let a2 = _mm256_loadu_ps(a_ptr.add(i + 16));
        let b2 = _mm256_loadu_ps(b_ptr.add(i + 16));
        acc2 = _mm256_fmadd_ps(a2, b2, acc2);

        let a3 = _mm256_loadu_ps(a_ptr.add(i + 24));
        let b3 = _mm256_loadu_ps(b_ptr.add(i + 24));
        acc3 = _mm256_fmadd_ps(a3, b3, acc3);

        i += 32;
    }

    // Handle 8-float chunks in remainder
    let remaining = n - i;
    let chunks8 = remaining / 8;
    for _ in 0..chunks8 {
        let va = _mm256_loadu_ps(a_ptr.add(i));
        let vb = _mm256_loadu_ps(b_ptr.add(i));
        acc0 = _mm256_fmadd_ps(va, vb, acc0);
        i += 8;
    }

    // Combine 4 accumulators
    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);

    // Horizontal sum of 8 floats in acc0
    let hi = _mm256_extractf128_ps(acc0, 1);
    let lo = _mm256_castps256_ps128(acc0);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    let mut total = _mm_cvtss_f32(result);

    // Scalar remainder (< 8 floats)
    while i < n {
        total += *a.get_unchecked(i) * *b.get_unchecked(i);
        i += 1;
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn l2_distance_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    let chunks = n / 16;
    let mut i = 0usize;

    for _ in 0..chunks {
        let a0 = _mm256_loadu_ps(a_ptr.add(i));
        let b0 = _mm256_loadu_ps(b_ptr.add(i));
        let d0 = _mm256_sub_ps(a0, b0);
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);

        let a1 = _mm256_loadu_ps(a_ptr.add(i + 8));
        let b1 = _mm256_loadu_ps(b_ptr.add(i + 8));
        let d1 = _mm256_sub_ps(a1, b1);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);

        i += 16;
    }

    acc0 = _mm256_add_ps(acc0, acc1);

    // Horizontal sum
    let hi = _mm256_extractf128_ps(acc0, 1);
    let lo = _mm256_castps256_ps128(acc0);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    let mut total = _mm_cvtss_f32(result);

    while i < n {
        let d = *a.get_unchecked(i) - *b.get_unchecked(i);
        total += d * d;
        i += 1;
    }

    total
}

// ==========================================================================
// Scalar fallback (ARM NEON will auto-vectorize these)
// ==========================================================================

fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
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
    for j in 0..remainder {
        unsafe { sum0 += *a.get_unchecked(i + j) * *b.get_unchecked(i + j); }
    }
    sum0 + sum1 + sum2 + sum3
}

fn l2_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut sum0: f32 = 0.0;
    let mut sum1: f32 = 0.0;
    let chunks = n / 4;
    let remainder = n % 4;
    let mut i = 0;

    for _ in 0..chunks {
        unsafe {
            let d0 = *a.get_unchecked(i) - *b.get_unchecked(i);
            let d1 = *a.get_unchecked(i + 1) - *b.get_unchecked(i + 1);
            let d2 = *a.get_unchecked(i + 2) - *b.get_unchecked(i + 2);
            let d3 = *a.get_unchecked(i + 3) - *b.get_unchecked(i + 3);
            sum0 += d0 * d0 + d2 * d2;
            sum1 += d1 * d1 + d3 * d3;
        }
        i += 4;
    }
    for j in 0..remainder {
        unsafe {
            let d = *a.get_unchecked(i + j) - *b.get_unchecked(i + j);
            sum0 += d * d;
        }
    }
    sum0 + sum1
}

// ==========================================================================
// Public API — dispatches to SIMD or scalar
// ==========================================================================

/// Dot product — selects the fastest available implementation.
/// On x86_64 with AVX2+FMA, this compiles to a direct call (no runtime check)
/// when built with target-cpu=native. The is_x86_feature_detected! macro
/// is optimized by LLVM to a constant when the feature is known at compile time.
#[inline(always)]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        // When compiled with -C target-cpu=native on AVX2+FMA hardware,
        // the compiler knows these features are available and eliminates the check.
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_product_avx2_fma(a, b) };
        }
    }

    dot_product_scalar(a, b)
}

/// Function pointer type for distance computation.
/// Set once at index construction time to eliminate per-call dispatch.
pub type DistanceFn = fn(&[f32], &[f32]) -> f32;

/// Get the optimal distance function pointer for a metric.
/// Call once at construction time, then use the pointer directly.
pub fn get_distance_fn(metric: DistanceMetric) -> DistanceFn {
    match metric {
        DistanceMetric::Cosine => cosine_distance,
        DistanceMetric::L2 => l2_distance,
        DistanceMetric::InnerProduct => inner_product_distance,
        DistanceMetric::Manhattan => manhattan_distance,
        DistanceMetric::Hamming => hamming_distance,
    }
}

#[inline(always)]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - dot_product(a, b)
}

#[inline(always)]
pub fn inner_product_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - dot_product(a, b)
}

#[inline(always)]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { l2_distance_avx2_fma(a, b) };
        }
    }

    l2_distance_scalar(a, b)
}

#[inline(always)]
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut sum0: f32 = 0.0;
    let mut sum1: f32 = 0.0;
    let chunks = n / 4;
    let remainder = n % 4;
    let mut i = 0;
    for _ in 0..chunks {
        unsafe {
            sum0 += (*a.get_unchecked(i) - *b.get_unchecked(i)).abs();
            sum1 += (*a.get_unchecked(i + 1) - *b.get_unchecked(i + 1)).abs();
            sum0 += (*a.get_unchecked(i + 2) - *b.get_unchecked(i + 2)).abs();
            sum1 += (*a.get_unchecked(i + 3) - *b.get_unchecked(i + 3)).abs();
        }
        i += 4;
    }
    for j in 0..remainder {
        unsafe { sum0 += (*a.get_unchecked(i + j) - *b.get_unchecked(i + j)).abs(); }
    }
    sum0 + sum1
}

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
/// Query loaded once, broadcast across 4 candidate vectors.
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

#[inline(always)]
pub fn cosine_distance_batch4(
    query: &[f32],
    v0: &[f32], v1: &[f32], v2: &[f32], v3: &[f32],
) -> (f32, f32, f32, f32) {
    let (d0, d1, d2, d3) = dot_product_batch4(query, v0, v1, v2, v3);
    (1.0 - d0, 1.0 - d1, 1.0 - d2, 1.0 - d3)
}

/// Prefetch hint for the next vector during graph traversal.
#[inline(always)]
pub fn prefetch_vector(ptr: *const f32) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        #[cfg(target_feature = "sse")]
        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
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
        assert!((l2_distance(&a, &b) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_manhattan_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 0.0, 1.0];
        assert!((manhattan_distance(&a, &b) - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_large() {
        let n = 131;
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.01).collect();
        let naive: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!((dot_product(&a, &b) - naive).abs() < 1e-2);
    }

    #[test]
    fn test_dot_product_dim128() {
        // Test with SIFT dimension — exercises the 32-wide main loop perfectly (128/32=4)
        let n = 128;
        let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).cos()).collect();
        let naive: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!((dot_product(&a, &b) - naive).abs() < 1e-3,
            "dot_product={} naive={}", dot_product(&a, &b), naive);
    }
}
