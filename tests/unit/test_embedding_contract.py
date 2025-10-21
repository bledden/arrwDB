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

    def test_validate_vectors_batch_all_valid(self, embedding_contract: LibraryEmbeddingContract, sample_vectors: list):
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

    def test_validate_vectors_batch_with_invalid_dimension(self, embedding_contract: LibraryEmbeddingContract, sample_vectors: list, vector_dimension: int):
        """Test that batch validation fails with wrong dimension."""
        vectors_list = [v.tolist() for v in sample_vectors]
        # Add one wrong dimension vector
        vectors_list.append(np.random.randn(vector_dimension + 5).tolist())

        # This will fail during numpy array conversion due to inhomogeneous shape
        with pytest.raises(ValueError, match="Failed to convert vectors|inhomogeneous"):
            embedding_contract.validate_vectors_batch(vectors_list)

    def test_validate_vectors_batch_with_nan(self, embedding_contract: LibraryEmbeddingContract, sample_vectors: list):
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

    def test_validate_vectors_batch_wrong_shape_1d(self, embedding_contract: LibraryEmbeddingContract):
        """Test that 1D array raises ValueError (line 130)."""
        vectors = np.random.randn(10)  # 1D array instead of 2D

        with pytest.raises(ValueError) as exc_info:
            embedding_contract.validate_vectors_batch(vectors.tolist())

        assert "2d array" in str(exc_info.value).lower()

    def test_validate_vectors_batch_dimension_mismatch(self, embedding_contract: LibraryEmbeddingContract, vector_dimension: int):
        """Test batch dimension mismatch (line 135)."""
        # Create batch with wrong dimension
        wrong_vectors = np.random.randn(5, vector_dimension + 10)

        with pytest.raises(ValueError) as exc_info:
            embedding_contract.validate_vectors_batch(wrong_vectors.tolist())

        assert "dimension mismatch" in str(exc_info.value).lower()

    def test_validate_vectors_batch_with_inf(self, embedding_contract: LibraryEmbeddingContract, sample_vectors: list):
        """Test batch with Inf values raises error (line 145)."""
        vectors = np.array(sample_vectors, dtype=np.float32)
        vectors[1, 0] = np.inf  # Add Inf to second vector

        with pytest.raises(ValueError) as exc_info:
            embedding_contract.validate_vectors_batch(vectors.tolist())

        assert "inf" in str(exc_info.value).lower()

    def test_validate_vectors_batch_with_zero_vector(self, embedding_contract: LibraryEmbeddingContract, vector_dimension: int):
        """Test batch with zero vector raises error (line 152-153)."""
        vectors = [
            np.random.randn(vector_dimension).tolist(),
            [0.0] * vector_dimension,  # Zero vector
            np.random.randn(vector_dimension).tolist(),
        ]

        with pytest.raises(ValueError) as exc_info:
            embedding_contract.validate_vectors_batch(vectors)

        error_msg = str(exc_info.value).lower()
        assert "zero vector" in error_msg
        assert "indices" in error_msg

    def test_repr(self, embedding_contract: LibraryEmbeddingContract, vector_dimension: int):
        """Test __repr__ method (line 163)."""
        repr_str = repr(embedding_contract)
        assert "LibraryEmbeddingContract" in repr_str
        assert str(vector_dimension) in repr_str

    def test_eq_same_dimension(self, vector_dimension: int):
        """Test equality with same dimension (line 167-169)."""
        contract1 = LibraryEmbeddingContract(expected_dimension=vector_dimension)
        contract2 = LibraryEmbeddingContract(expected_dimension=vector_dimension)
        assert contract1 == contract2

    def test_eq_different_dimension(self, vector_dimension: int):
        """Test inequality with different dimension (line 167-169)."""
        contract1 = LibraryEmbeddingContract(expected_dimension=vector_dimension)
        contract2 = LibraryEmbeddingContract(expected_dimension=vector_dimension + 10)
        assert contract1 != contract2

    def test_eq_non_contract_object(self, embedding_contract: LibraryEmbeddingContract):
        """Test equality with non-contract object (line 167-169)."""
        assert embedding_contract != "not a contract"
        assert embedding_contract != 123
        assert embedding_contract != None

    def test_hash(self, embedding_contract: LibraryEmbeddingContract, vector_dimension: int):
        """Test __hash__ method (line 173)."""
        hash_value = hash(embedding_contract)
        assert isinstance(hash_value, int)

        # Same dimension should have same hash
        contract2 = LibraryEmbeddingContract(expected_dimension=vector_dimension)
        assert hash(embedding_contract) == hash(contract2)

        # Can be used in set
        contract_set = {embedding_contract, contract2}
        assert len(contract_set) == 1  # Should be deduplicated
