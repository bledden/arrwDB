"""
Unit tests for LibraryEmbeddingContract.
"""

import pytest
import numpy as np

from core.embedding_contract import LibraryEmbeddingContract


@pytest.mark.unit
class TestEmbeddingContractInitialization:
    """Tests for EmbeddingContract initialization."""

    def test_valid_initialization(self, vector_dimension: int):
        """Test valid initialization."""
        contract = LibraryEmbeddingContract(expected_dimension=vector_dimension)
        assert contract.dimension == vector_dimension

    def test_invalid_dimension_raises_error(self):
        """Test that invalid dimensions raise ValueError."""
        with pytest.raises(ValueError, match="Expected dimension must be positive"):
            LibraryEmbeddingContract(expected_dimension=0)

        with pytest.raises(ValueError, match="Expected dimension must be positive"):
            LibraryEmbeddingContract(expected_dimension=-10)


@pytest.mark.unit
class TestValidateVector:
    """Tests for vector validation."""

    def test_validate_correct_dimension(self, embedding_contract: LibraryEmbeddingContract, vector_dimension: int):
        """Test validating a vector with correct dimension."""
        vector = np.random.randn(vector_dimension).tolist()
        validated = embedding_contract.validate_vector(vector)

        assert validated.shape == (vector_dimension,)
        assert validated.dtype == np.float32
        # Should be normalized
        assert np.abs(np.linalg.norm(validated) - 1.0) < 1e-5

    def test_validate_wrong_dimension_raises_error(self, embedding_contract: LibraryEmbeddingContract, vector_dimension: int):
        """Test that wrong dimension raises ValueError."""
        wrong_vector = np.random.randn(vector_dimension + 10).tolist()

        with pytest.raises(ValueError, match="dimension mismatch"):
            embedding_contract.validate_vector(wrong_vector)

    def test_validate_nan_raises_error(self, embedding_contract: LibraryEmbeddingContract, vector_dimension: int):
        """Test that NaN values raise ValueError."""
        vector = np.random.randn(vector_dimension)
        vector[0] = np.nan

        with pytest.raises(ValueError, match="NaN|invalid"):
            embedding_contract.validate_vector(vector.tolist())

    def test_validate_inf_raises_error(self, embedding_contract: LibraryEmbeddingContract, vector_dimension: int):
        """Test that Inf values raise ValueError."""
        vector = np.random.randn(vector_dimension)
        vector[0] = np.inf

        with pytest.raises(ValueError, match="Inf|invalid"):
            embedding_contract.validate_vector(vector.tolist())

    def test_validate_zero_vector_raises_error(self, embedding_contract: LibraryEmbeddingContract, vector_dimension: int):
        """Test that zero vector raises ValueError."""
        zero_vector = np.zeros(vector_dimension).tolist()

        with pytest.raises(ValueError, match="zero|norm"):
            embedding_contract.validate_vector(zero_vector)

    def test_normalization(self, embedding_contract: LibraryEmbeddingContract, vector_dimension: int):
        """Test that vectors are normalized to unit length."""
        # Create non-normalized vector
        vector = np.random.randn(vector_dimension) * 10  # Scale up

        validated = embedding_contract.validate_vector(vector.tolist())

        # Should be normalized to unit length
        norm = np.linalg.norm(validated)
        assert np.abs(norm - 1.0) < 1e-5


@pytest.mark.unit
class TestValidateBatch:
    """Tests for batch validation."""

    def test_validate_batch_all_valid(self, embedding_contract: LibraryEmbeddingContract, sample_vectors: list):
        """Test validating a batch of valid vectors."""
        vectors_list = [v.tolist() for v in sample_vectors]
        validated = embedding_contract.validate_vectors_batch(vectors_list)

        assert len(validated) == len(sample_vectors)
        for vec in validated:
            assert vec.dtype == np.float32
            assert np.abs(np.linalg.norm(vec) - 1.0) < 1e-5

    def test_validate_empty_batch(self, embedding_contract: LibraryEmbeddingContract):
        """Test validating an empty batch raises error."""
        with pytest.raises(ValueError, match="Cannot validate empty batch"):
            embedding_contract.validate_vectors_batch([])

    def test_validate_batch_with_invalid_dimension(self, embedding_contract: LibraryEmbeddingContract, sample_vectors: list, vector_dimension: int):
        """Test that batch validation fails with wrong dimension."""
        vectors_list = [v.tolist() for v in sample_vectors]
        # Add one wrong dimension vector
        vectors_list.append(np.random.randn(vector_dimension + 5).tolist())

        # This will fail during numpy array conversion due to inhomogeneous shape
        with pytest.raises(ValueError, match="Failed to convert vectors|inhomogeneous"):
            embedding_contract.validate_vectors_batch(vectors_list)

    def test_validate_batch_with_nan(self, embedding_contract: LibraryEmbeddingContract, sample_vectors: list):
        """Test that batch validation fails with NaN."""
        vectors_list = [v.tolist() for v in sample_vectors]
        # Add NaN vector
        bad_vec = sample_vectors[0].copy()
        bad_vec[0] = np.nan
        vectors_list.append(bad_vec.tolist())

        with pytest.raises(ValueError, match="NaN"):
            embedding_contract.validate_vectors_batch(vectors_list)


@pytest.mark.unit
class TestEmbeddingContractEdgeCases:
    """Edge case tests for EmbeddingContract."""

    def test_very_small_vector(self, embedding_contract: LibraryEmbeddingContract, vector_dimension: int):
        """Test validating very small but non-zero vector."""
        tiny_vector = np.full(vector_dimension, 1e-10)

        validated = embedding_contract.validate_vector(tiny_vector.tolist())

        # Should still normalize correctly
        assert np.abs(np.linalg.norm(validated) - 1.0) < 1e-5

    def test_single_element_dimension(self):
        """Test contract with dimension 1."""
        contract = LibraryEmbeddingContract(expected_dimension=1)
        vector = [5.0]

        validated = contract.validate_vector(vector)

        assert validated.shape == (1,)
        assert np.abs(validated[0] - 1.0) < 1e-5  # Should be normalized to 1.0

    def test_large_dimension(self):
        """Test contract with very large dimension."""
        dimension = 4096
        contract = LibraryEmbeddingContract(expected_dimension=dimension)

        vector = np.random.randn(dimension)
        validated = contract.validate_vector(vector.tolist())

        assert validated.shape == (dimension,)
        assert np.abs(np.linalg.norm(validated) - 1.0) < 1e-5
