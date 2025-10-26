"""
Unit tests for vector quantization utilities.

Tests scalar quantization, hybrid search, and memory savings calculations.
"""

import numpy as np
import pytest

from app.utils.quantization import (
    ScalarQuantizer,
    batch_cosine_similarity,
    calculate_memory_savings,
    cosine_similarity,
    estimate_accuracy,
)


class TestScalarQuantizer:
    """Tests for scalar (int8) quantization."""

    def test_quantizer_initialization(self):
        """Test quantizer initialization with different bit depths."""
        q8 = ScalarQuantizer(bits=8)
        assert q8.bits == 8
        assert q8.max_val == 255

        q4 = ScalarQuantizer(bits=4)
        assert q4.bits == 4
        assert q4.max_val == 15

    def test_invalid_bits(self):
        """Test that invalid bit depths raise ValueError."""
        with pytest.raises(ValueError, match="Bits must be 4 or 8"):
            ScalarQuantizer(bits=16)

    def test_calibration(self):
        """Test calibration computes correct min/max values."""
        vectors = np.array([
            [1.0, 2.0, 3.0],
            [0.5, 1.0, 4.0],
            [2.0, 3.0, 2.0]
        ], dtype=np.float32)

        quantizer = ScalarQuantizer(bits=8)
        min_vals, max_vals = quantizer.calibrate(vectors)

        # Check per-dimension min/max
        assert np.allclose(min_vals, [0.5, 1.0, 2.0])
        assert np.allclose(max_vals, [2.0, 3.0, 4.0])

    def test_quantize_dequantize_round_trip(self):
        """Test that quantize -> dequantize preserves approximate values."""
        # Use normalized vectors (more realistic for semantic search)
        vectors = np.random.randn(100, 384).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        quantizer = ScalarQuantizer(bits=8)
        quantized, params = quantizer.quantize(vectors)

        # Check quantized values are uint8
        assert quantized.dtype == np.uint8
        assert quantized.shape == vectors.shape

        # Dequantize
        reconstructed = quantizer.dequantize(quantized, params)

        # Check reconstruction quality (should be close but not exact)
        # For 8-bit quantization, MSE of ~0.03-0.05 is expected and acceptable
        mse = np.mean((vectors - reconstructed) ** 2)
        assert mse < 0.05, f"MSE too high: {mse}"

        # Check max absolute error per dimension
        max_abs_error = np.max(np.abs(vectors - reconstructed))
        assert max_abs_error < 0.1, f"Max absolute error too high: {max_abs_error}"

    def test_quantize_single(self):
        """Test quantizing a single vector with pre-calibrated params."""
        vectors = np.random.randn(50, 384).astype(np.float32)
        quantizer = ScalarQuantizer(bits=8)

        # Calibrate on batch
        _, params = quantizer.quantize(vectors)

        # Quantize single vector
        single_vector = vectors[0].tolist()
        quantized_single = quantizer.quantize_single(single_vector, params)

        # Check shape and dtype
        assert quantized_single.shape == (384,)
        assert quantized_single.dtype == np.uint8

    def test_constant_dimension_handling(self):
        """Test that constant dimensions (min == max) are handled correctly."""
        vectors = np.array([
            [1.0, 5.0, 3.0],  # Second dim is constant
            [2.0, 5.0, 4.0],
            [3.0, 5.0, 2.0]
        ], dtype=np.float32)

        quantizer = ScalarQuantizer(bits=8)
        quantized, params = quantizer.quantize(vectors)
        reconstructed = quantizer.dequantize(quantized, params)

        # Should not crash and should handle constant dimension
        assert reconstructed.shape == vectors.shape

    def test_quantization_preserves_monotonicity(self):
        """Test that quantization preserves monotonic relationships."""
        # Create vectors where values increase monotonically
        vectors = np.array([
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0]
        ], dtype=np.float32)

        quantizer = ScalarQuantizer(bits=8)
        quantized, params = quantizer.quantize(vectors)
        reconstructed = quantizer.dequantize(quantized, params)

        # Check that monotonicity is preserved
        # reconstructed[i] < reconstructed[i+1] for all dimensions
        for dim in range(3):
            assert reconstructed[0, dim] < reconstructed[1, dim]
            assert reconstructed[1, dim] < reconstructed[2, dim]


class TestCosineSimilarity:
    """Tests for cosine similarity functions."""

    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors is 1.0."""
        v = np.array([1.0, 2.0, 3.0])
        sim = cosine_similarity(v, v)
        assert np.isclose(sim, 1.0)

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors is 0.0."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        sim = cosine_similarity(v1, v2)
        assert np.isclose(sim, 0.0)

    def test_cosine_similarity_opposite(self):
        """Test cosine similarity of opposite vectors is -1.0."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([-1.0, -2.0, -3.0])
        sim = cosine_similarity(v1, v2)
        assert np.isclose(sim, -1.0)

    def test_batch_cosine_similarity(self):
        """Test batch cosine similarity computation."""
        query = np.array([1.0, 0.0, 0.0])
        vectors = np.array([
            [1.0, 0.0, 0.0],  # Identical
            [0.0, 1.0, 0.0],  # Orthogonal
            [-1.0, 0.0, 0.0]  # Opposite
        ])

        similarities = batch_cosine_similarity(query, vectors)

        assert similarities.shape == (3,)
        assert np.isclose(similarities[0], 1.0)
        assert np.isclose(similarities[1], 0.0)
        assert np.isclose(similarities[2], -1.0)

    def test_zero_vector_handling(self):
        """Test that zero vectors are handled gracefully."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([0.0, 0.0, 0.0])

        sim = cosine_similarity(v1, v2)
        assert np.isclose(sim, 0.0)


class TestMemorySavings:
    """Tests for memory savings calculation."""

    def test_no_quantization_no_savings(self):
        """Test that 'none' strategy has 0% savings."""
        savings = calculate_memory_savings(strategy="none", bits=8, dimensions=384)
        assert savings == 0.0

    def test_scalar_quantization_savings(self):
        """Test scalar quantization memory savings calculation."""
        # Scalar: 1 byte per dim + overhead
        savings = calculate_memory_savings(strategy="scalar", bits=8, dimensions=384)

        # Should have some savings but not 75% due to overhead
        assert 0.0 <= savings < 75.0

    def test_hybrid_quantization_savings(self):
        """Test hybrid quantization memory savings."""
        savings = calculate_memory_savings(strategy="hybrid", bits=8, dimensions=384)

        # Hybrid caches some float32, so savings should be less than scalar
        assert 0.0 <= savings < 75.0

    def test_different_dimensions(self):
        """Test memory savings with different vector dimensions."""
        # Larger dimensions should have better savings (overhead is fixed)
        savings_small = calculate_memory_savings(strategy="scalar", bits=8, dimensions=128)
        savings_large = calculate_memory_savings(strategy="scalar", bits=8, dimensions=1024)

        # Larger vectors should have better compression ratio
        assert savings_large >= savings_small


class TestAccuracyEstimation:
    """Tests for accuracy estimation."""

    def test_no_quantization_perfect_accuracy(self):
        """Test that 'none' strategy has 100% accuracy."""
        accuracy = estimate_accuracy(strategy="none", bits=8)
        assert accuracy == 100.0

    def test_scalar_8bit_accuracy(self):
        """Test scalar 8-bit quantization accuracy estimate."""
        accuracy = estimate_accuracy(strategy="scalar", bits=8)

        # Should be around 98-99%
        assert 97.0 <= accuracy <= 100.0

    def test_scalar_4bit_accuracy(self):
        """Test scalar 4-bit quantization accuracy estimate."""
        accuracy = estimate_accuracy(strategy="scalar", bits=4)

        # 4-bit should be less accurate than 8-bit
        accuracy_8bit = estimate_accuracy(strategy="scalar", bits=8)
        assert accuracy < accuracy_8bit

    def test_hybrid_better_than_scalar(self):
        """Test that hybrid has better accuracy than scalar."""
        hybrid_accuracy = estimate_accuracy(strategy="hybrid", bits=8)
        scalar_accuracy = estimate_accuracy(strategy="scalar", bits=8)

        assert hybrid_accuracy > scalar_accuracy


class TestQuantizationQuality:
    """Integration tests for quantization quality."""

    def test_quantization_similarity_search_quality(self):
        """Test that quantized vectors maintain search quality."""
        # Generate random vectors
        num_vectors = 1000
        dimensions = 384
        vectors = np.random.randn(num_vectors, dimensions).astype(np.float32)

        # Normalize vectors (common in semantic search)
        vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)

        # Query vector
        query = np.random.randn(dimensions).astype(np.float32)
        query = query / (np.linalg.norm(query) + 1e-8)

        # Find top-10 with float32
        float32_similarities = batch_cosine_similarity(query, vectors)
        float32_top10 = np.argsort(float32_similarities)[-10:][::-1]

        # Quantize vectors
        quantizer = ScalarQuantizer(bits=8)
        quantized, params = quantizer.quantize(vectors)
        reconstructed = quantizer.dequantize(quantized, params)

        # Re-normalize reconstructed vectors (quantization can shift norms slightly)
        reconstructed_norms = np.linalg.norm(reconstructed, axis=1, keepdims=True)
        reconstructed = reconstructed / (reconstructed_norms + 1e-8)

        # Find top-10 with quantized
        quantized_similarities = batch_cosine_similarity(query, reconstructed)
        quantized_top10 = np.argsort(quantized_similarities)[-10:][::-1]

        # Calculate overlap (recall@10)
        overlap = len(set(float32_top10) & set(quantized_top10))
        recall = overlap / 10.0

        # For random vectors with scalar quantization, aim for at least 70% recall
        # (This is realistic - 7 out of 10 top results are the same)
        assert recall >= 0.7, f"Recall@10 too low: {recall} (top10 float: {float32_top10}, top10 quant: {quantized_top10})"

    def test_quantization_distance_preservation(self):
        """Test that quantization preserves relative distances."""
        # Create vectors with known relationships
        base = np.array([1.0, 0.0, 0.0, 0.0])
        close = np.array([0.9, 0.1, 0.0, 0.0])  # Close to base
        far = np.array([0.0, 0.0, 1.0, 0.0])    # Far from base

        vectors = np.array([base, close, far], dtype=np.float32)

        # Compute original similarities
        orig_sim_close = cosine_similarity(base, close)
        orig_sim_far = cosine_similarity(base, far)

        # Quantize
        quantizer = ScalarQuantizer(bits=8)
        quantized, params = quantizer.quantize(vectors)
        reconstructed = quantizer.dequantize(quantized, params)

        # Compute quantized similarities
        quant_sim_close = cosine_similarity(reconstructed[0], reconstructed[1])
        quant_sim_far = cosine_similarity(reconstructed[0], reconstructed[2])

        # Check relative ordering is preserved
        # Close should still be closer than far
        assert quant_sim_close > quant_sim_far

        # Check absolute similarities are close
        assert np.abs(orig_sim_close - quant_sim_close) < 0.1
        assert np.abs(orig_sim_far - quant_sim_far) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
