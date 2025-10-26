"""
Vector quantization utilities for memory optimization.

Provides scalar (int8) and hybrid quantization strategies for reducing
memory usage while maintaining high search accuracy.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np


class ScalarQuantizer:
    """
    Scalar quantization using per-dimension min/max scaling to int8.

    Converts float32 vectors (4 bytes/dim) to int8 (1 byte/dim) for 4x memory savings.
    Typical accuracy loss: 1-2% recall@10.

    Example:
        quantizer = ScalarQuantizer(bits=8)
        quantized, params = quantizer.quantize(vectors)
        reconstructed = quantizer.dequantize(quantized, params)
    """

    def __init__(self, bits: int = 8):
        """
        Initialize scalar quantizer.

        Args:
            bits: Number of bits per dimension (4 or 8)
        """
        if bits not in [4, 8]:
            raise ValueError("Bits must be 4 or 8")

        self.bits = bits
        self.dtype = np.uint8  # Use unsigned int8 for 0-255 range
        self.max_val = (2 ** bits) - 1  # 255 for 8-bit, 15 for 4-bit

    def calibrate(self, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calibrate quantization parameters from sample vectors.

        Computes per-dimension min/max values for optimal scaling.

        Args:
            vectors: Shape (n_vectors, dimensions) float32 array

        Returns:
            (min_vals, max_vals): Per-dimension min and max values
        """
        min_vals = np.min(vectors, axis=0, keepdims=False).astype(np.float32)
        max_vals = np.max(vectors, axis=0, keepdims=False).astype(np.float32)

        # Handle edge case where min == max (constant dimension)
        # Add small epsilon to avoid division by zero
        range_vals = max_vals - min_vals
        mask = range_vals < 1e-8
        max_vals = np.where(mask, min_vals + 1e-6, max_vals)

        return min_vals, max_vals

    def quantize(
        self, vectors: np.ndarray, calibration_params: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Quantize float32 vectors to int8.

        Args:
            vectors: Shape (n_vectors, dimensions) float32 array
            calibration_params: Optional (min_vals, max_vals) from calibrate().
                               If None, calibrates on provided vectors.

        Returns:
            (quantized_vectors, params):
                - quantized_vectors: Shape (n_vectors, dimensions) int8 array
                - params: Dict with 'min_vals', 'max_vals', 'bits' for dequantization
        """
        if calibration_params is None:
            min_vals, max_vals = self.calibrate(vectors)
        else:
            min_vals, max_vals = calibration_params

        # Scale to [0, max_val] range
        range_vals = max_vals - min_vals
        range_vals = np.where(range_vals < 1e-8, 1.0, range_vals)  # Avoid division by zero

        scaled = (vectors - min_vals) / range_vals
        quantized = np.clip(scaled * self.max_val, 0, self.max_val).astype(self.dtype)

        params = {
            'min_vals': min_vals,
            'max_vals': max_vals,
            'bits': self.bits
        }

        return quantized, params

    def dequantize(self, quantized: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Reconstruct float32 vectors from quantized int8.

        Args:
            quantized: Shape (n_vectors, dimensions) int8 array
            params: Dict with 'min_vals', 'max_vals', 'bits'

        Returns:
            reconstructed: Shape (n_vectors, dimensions) float32 array
        """
        min_vals = params['min_vals']
        max_vals = params['max_vals']
        bits = params['bits']
        max_val = (2 ** bits) - 1

        # Scale back to original range
        range_vals = max_vals - min_vals
        scaled = quantized.astype(np.float32) / max_val
        reconstructed = scaled * range_vals + min_vals

        return reconstructed

    def quantize_single(
        self, vector: List[float], params: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Quantize a single vector using pre-calibrated parameters.

        Args:
            vector: Single vector as list of floats
            params: Dict with 'min_vals', 'max_vals', 'bits'

        Returns:
            quantized: int8 array
        """
        vector_arr = np.array(vector, dtype=np.float32).reshape(1, -1)
        min_vals = params['min_vals']
        max_vals = params['max_vals']

        range_vals = max_vals - min_vals
        range_vals = np.where(range_vals < 1e-8, 1.0, range_vals)

        scaled = (vector_arr - min_vals) / range_vals
        quantized = np.clip(scaled * self.max_val, 0, self.max_val).astype(self.dtype)

        return quantized[0]


def calculate_memory_savings(
    strategy: str, bits: int = 8, dimensions: int = 384
) -> float:
    """
    Calculate memory savings percentage for a quantization strategy.

    Args:
        strategy: "none", "scalar", "hybrid"
        bits: Number of bits (4 or 8)
        dimensions: Vector dimensions

    Returns:
        Memory savings as percentage (0-100)
    """
    float32_bytes = dimensions * 4  # 4 bytes per dimension

    if strategy == "none":
        return 0.0
    elif strategy == "scalar":
        quantized_bytes = dimensions * 1  # 1 byte per dimension (int8)
        # Add overhead for min/max parameters (2 float32 per dimension)
        overhead_bytes = dimensions * 2 * 4
        total_bytes = quantized_bytes + overhead_bytes
        savings = (1 - total_bytes / float32_bytes) * 100
        return max(0.0, savings)
    elif strategy == "hybrid":
        # Hybrid stores both quantized and float32 for top-K reranking
        # Assume 10% float32 cache for hot vectors
        quantized_bytes = dimensions * 1
        cache_bytes = dimensions * 4 * 0.1  # 10% cached
        overhead_bytes = dimensions * 2 * 4
        total_bytes = quantized_bytes + cache_bytes + overhead_bytes
        savings = (1 - total_bytes / float32_bytes) * 100
        return max(0.0, savings)
    else:
        return 0.0


def estimate_accuracy(strategy: str, bits: int = 8) -> float:
    """
    Estimate accuracy retention for a quantization strategy.

    Based on research and benchmarks for typical semantic search tasks.

    Args:
        strategy: "none", "scalar", "hybrid"
        bits: Number of bits (4 or 8)

    Returns:
        Estimated accuracy percentage (0-100)
    """
    if strategy == "none":
        return 100.0
    elif strategy == "scalar":
        if bits == 8:
            return 98.5  # ~1.5% recall loss
        elif bits == 4:
            return 95.0  # ~5% recall loss
        else:
            return 97.0
    elif strategy == "hybrid":
        # Hybrid reranks top-K with float32, so accuracy is very high
        return 99.5  # ~0.5% recall loss
    else:
        return 100.0


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: Vector 1
        b: Vector 2

    Returns:
        Cosine similarity in range [-1, 1]
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0

    return dot_product / (norm_a * norm_b)


def batch_cosine_similarity(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query and multiple vectors.

    Args:
        query: Shape (dimensions,)
        vectors: Shape (n_vectors, dimensions)

    Returns:
        similarities: Shape (n_vectors,) with cosine similarities
    """
    # Normalize query
    query_norm = query / (np.linalg.norm(query) + 1e-8)

    # Normalize vectors
    vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)

    # Compute dot products
    similarities = np.dot(vectors_norm, query_norm)

    return similarities
