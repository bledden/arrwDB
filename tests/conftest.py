"""
Pytest configuration and shared fixtures.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from uuid import uuid4, UUID
from typing import Generator
import numpy as np

from app.models.base import Library, Document, Chunk, LibraryMetadata, DocumentMetadata, ChunkMetadata
from core.vector_store import VectorStore
from core.embedding_contract import LibraryEmbeddingContract
from infrastructure.repositories.library_repository import LibraryRepository
from infrastructure.indexes.brute_force import BruteForceIndex
from infrastructure.indexes.kd_tree import KDTreeIndex
from infrastructure.indexes.lsh import LSHIndex
from infrastructure.indexes.hnsw import HNSWIndex


@pytest.fixture
def temp_data_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test data."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def vector_dimension() -> int:
    """Standard vector dimension for tests."""
    return 128


@pytest.fixture
def vector_store(vector_dimension: int, temp_data_dir: Path) -> VectorStore:
    """Create a VectorStore for testing."""
    return VectorStore(dimension=vector_dimension, initial_capacity=100)


@pytest.fixture
def embedding_contract(vector_dimension: int) -> LibraryEmbeddingContract:
    """Create an EmbeddingContract for testing."""
    return LibraryEmbeddingContract(expected_dimension=vector_dimension)


@pytest.fixture
def sample_vectors(vector_dimension: int) -> list:
    """Generate sample normalized vectors for testing."""
    np.random.seed(42)
    vectors = []
    for _ in range(10):
        vec = np.random.randn(vector_dimension).astype(np.float32)
        vec = vec / np.linalg.norm(vec)  # Normalize
        vectors.append(vec)
    return vectors


@pytest.fixture
def library_repository(temp_data_dir: Path) -> LibraryRepository:
    """Create a LibraryRepository for testing."""
    return LibraryRepository(data_dir=temp_data_dir)


@pytest.fixture
def sample_library(vector_dimension: int) -> Library:
    """Create a sample library for testing."""
    metadata = LibraryMetadata(
        description="Test library",
        index_type="brute_force",
        embedding_dimension=vector_dimension,
        embedding_model="test-model-v1",
    )
    return Library(name="Test Library", metadata=metadata)


@pytest.fixture
def sample_document(vector_dimension: int) -> Document:
    """Create a sample document for testing."""
    doc_id = uuid4()

    # Create a chunk with proper metadata
    vec = np.random.randn(vector_dimension).astype(np.float32)
    vec = vec / np.linalg.norm(vec)

    chunk_metadata = ChunkMetadata(
        chunk_index=0,
        source_document_id=doc_id,
    )

    chunk = Chunk(
        text="This is a test document with some content.",
        embedding=vec.tolist(),
        metadata=chunk_metadata,
    )

    doc_metadata = DocumentMetadata(
        title="Test Document",
        author="Test Author",
    )

    return Document(
        id=doc_id,
        chunks=[chunk],
        metadata=doc_metadata,
    )


@pytest.fixture
def sample_chunk(vector_dimension: int) -> Chunk:
    """Create a sample chunk with embedding for testing."""
    vec = np.random.randn(vector_dimension).astype(np.float32)
    vec = vec / np.linalg.norm(vec)  # Normalize

    metadata = ChunkMetadata(
        chunk_index=0,
        source_document_id=uuid4(),
    )

    return Chunk(
        text="This is a test chunk.",
        embedding=vec.tolist(),
        metadata=metadata,
    )


@pytest.fixture(params=["brute_force", "kd_tree", "lsh", "hnsw"])
def index_type(request) -> str:
    """Parametrized fixture for all index types."""
    return request.param


@pytest.fixture
def create_index(vector_store: VectorStore):
    """Factory fixture for creating indexes."""
    def _create(index_type: str):
        if index_type == "brute_force":
            return BruteForceIndex(vector_store)
        elif index_type == "kd_tree":
            return KDTreeIndex(vector_store)
        elif index_type == "lsh":
            return LSHIndex(vector_store, num_tables=5, hash_size=10)
        elif index_type == "hnsw":
            return HNSWIndex(vector_store, M=16, ef_construction=200)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
    return _create
