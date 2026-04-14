/// Distance metric selection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DistanceMetric {
    Cosine,
    L2,
    InnerProduct,
}

/// Compute distance between two vectors using the specified metric.
#[inline(always)]
pub fn compute_distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::Cosine => cosine_distance(a, b),
        DistanceMetric::L2 => l2_distance(a, b),
        DistanceMetric::InnerProduct => inner_product_distance(a, b),
    }
}

/// Cosine distance = 1 - dot_product. Assumes normalized vectors.
#[inline(always)]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have same length");
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    1.0 - dot
}

/// Squared L2 (Euclidean) distance. No sqrt for performance — ranking is preserved.
#[inline(always)]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have same length");
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Inner product distance = 1 - dot_product. Smaller = more similar.
#[inline(always)]
pub fn inner_product_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have same length");
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    1.0 - dot
}

/// Dot product (raw, no distance conversion).
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have same length");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
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
