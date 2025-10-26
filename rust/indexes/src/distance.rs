/// Compute cosine distance between two vectors.
///
/// Cosine distance = 1 - cosine_similarity = 1 - (dot_product)
/// Assumes vectors are already normalized (which they should be for embeddings).
///
/// This implementation will automatically use SIMD instructions where available
/// thanks to Rust's auto-vectorization and the explicit use of iterators.
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have same length");

    // Compute dot product using SIMD-friendly iterator pattern
    let dot_product: f32 = a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum();

    // Return cosine distance
    1.0 - dot_product
}

/// Compute dot product between two vectors (SIMD-optimized).
///
/// This is used internally for cosine similarity calculations.
/// The compiler will auto-vectorize this loop when possible.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have same length");

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_distance_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let dist = cosine_distance(&a, &b);
        assert!((dist - 0.0).abs() < 1e-6, "Identical vectors should have distance ~0");
    }

    #[test]
    fn test_cosine_distance_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let dist = cosine_distance(&a, &b);
        assert!((dist - 1.0).abs() < 1e-6, "Orthogonal vectors should have distance ~1");
    }

    #[test]
    fn test_cosine_distance_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let dist = cosine_distance(&a, &b);
        assert!((dist - 2.0).abs() < 1e-6, "Opposite vectors should have distance ~2");
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = dot_product(&a, &b);
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!((result - 32.0).abs() < 1e-6);
    }
}
