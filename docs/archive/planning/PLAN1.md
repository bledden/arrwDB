# Vector Database REST API - Master Implementation Plan

## Project Overview
Build a production-grade REST API for a Vector Database that indexes and queries documents using custom-implemented indexing algorithms. The system must demonstrate exceptional code quality, architectural design, and scalability considerations.

**Tech Stack:** Python, FastAPI, Pydantic, NumPy, Docker, Temporal (optional)

**Key Constraint:** No external vector DB libraries (FAISS, Pinecone, ChromaDB, etc.) - implement algorithms from scratch

---

## Enhanced Architecture & Design Principles

### 1. Domain-Driven Design (DDD) Structure
```
src/
├── domain/
│   ├── models/          # Pydantic domain models (Chunk, Document, Library)
│   ├── value_objects/   # Immutable value objects (EmbeddingVector, ChunkId, etc.)
│   ├── entities/        # Business entities with identity
│   ├── aggregates/      # Aggregate roots (Library as aggregate root)
│   └── exceptions/      # Domain-specific exceptions
├── application/
│   ├── services/        # Application services (LibraryService, QueryService)
│   ├── dtos/            # Data Transfer Objects for API I/O
│   └── use_cases/       # Specific use case implementations
├── infrastructure/
│   ├── repositories/    # Repository implementations (InMemoryRepo, FileRepo)
│   ├── indexes/         # Vector index implementations
│   ├── persistence/     # Disk persistence layer
│   └── external/        # External API clients (Cohere embedding)
├── api/
│   ├── routes/          # FastAPI routers
│   ├── dependencies/    # FastAPI dependencies
│   ├── middleware/      # Custom middleware
│   └── schemas/         # API request/response schemas (separate from domain)
└── temporal/            # Temporal workflows and activities (if implemented)
```

### 2. Core Design Patterns to Implement
- **Repository Pattern**: Abstract data access
- **Strategy Pattern**: Swappable index algorithms
- **Factory Pattern**: Index creation based on type
- **Singleton Pattern**: Global state management with thread safety
- **Command Pattern**: For operations that modify state (enables undo/redo)
- **Observer Pattern**: For replication in leader-follower architecture

---

## Phase 1: Foundation & Core Domain (Priority: CRITICAL)

### Task 1.1: Domain Model Design
**Deliverable:** Complete Pydantic models with full validation

**Implementation Details:**
```python
# Enhanced schema with strict validation
from pydantic import BaseModel, Field, validator, UUID4
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class ChunkMetadata(BaseModel):
    """Structured metadata for chunks"""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    source_page: Optional[int] = None
    language: str = "en"
    custom_fields: Dict[str, Any] = Field(default_factory=dict)

class Chunk(BaseModel):
    """Immutable chunk with embedding"""
    id: UUID4 = Field(default_factory=uuid4)
    text: str = Field(..., min_length=1, max_length=10000)
    embedding: List[float] = Field(..., min_items=1)
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)

    @validator('embedding')
    def validate_embedding_dimension(cls, v):
        if len(v) not in [384, 768, 1024, 1536]:  # Common dimensions
            raise ValueError(f"Embedding dimension {len(v)} not supported")
        return v

    class Config:
        frozen = True  # Immutability

class DocumentMetadata(BaseModel):
    title: str
    author: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    tags: List[str] = Field(default_factory=list)
    custom_fields: Dict[str, Any] = Field(default_factory=dict)

class Document(BaseModel):
    id: UUID4 = Field(default_factory=uuid4)
    chunks: List[Chunk] = Field(..., min_items=1)
    metadata: DocumentMetadata

class LibraryMetadata(BaseModel):
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    index_type: str = "brute_force"
    embedding_dimension: int = 768

class Library(BaseModel):
    id: UUID4 = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=255)
    documents: List[Document] = Field(default_factory=list)
    metadata: LibraryMetadata = Field(default_factory=LibraryMetadata)
```

**Key Improvements:**
- Separate metadata classes for better structure
- Validation for embedding dimensions
- Immutable chunks (frozen config)
- UUID4 for all IDs
- Timestamps on all entities
- Min/max length constraints

### Task 1.2: Custom Exception Hierarchy
```python
# domain/exceptions.py
from fastapi import status

class VectorDBException(Exception):
    """Base exception for all domain errors"""
    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class ResourceNotFoundError(VectorDBException):
    def __init__(self, resource: str, id: str):
        super().__init__(
            f"{resource} with id {id} not found",
            status.HTTP_404_NOT_FOUND
        )

class LibraryNotFoundError(ResourceNotFoundError):
    def __init__(self, library_id: str):
        super().__init__("Library", library_id)

class DocumentNotFoundError(ResourceNotFoundError):
    def __init__(self, document_id: str):
        super().__init__("Document", document_id)

class ChunkNotFoundError(ResourceNotFoundError):
    def __init__(self, chunk_id: str):
        super().__init__("Chunk", chunk_id)

class DuplicateResourceError(VectorDBException):
    def __init__(self, resource: str, id: str):
        super().__init__(
            f"{resource} with id {id} already exists",
            status.HTTP_409_CONFLICT
        )

class InvalidOperationError(VectorDBException):
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_400_BAD_REQUEST)

class IndexNotBuiltError(VectorDBException):
    def __init__(self, library_id: str):
        super().__init__(
            f"Index not built for library {library_id}. Call /index endpoint first.",
            status.HTTP_412_PRECONDITION_FAILED
        )

class EmbeddingDimensionMismatchError(VectorDBException):
    def __init__(self, expected: int, got: int):
        super().__init__(
            f"Embedding dimension mismatch: expected {expected}, got {got}",
            status.HTTP_400_BAD_REQUEST
        )
```

---

## Phase 2: Vector Index Algorithms (Priority: CRITICAL)

### Task 2.1: Index Interface & Base Class
```python
# infrastructure/indexes/base.py
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Protocol
import numpy as np
from dataclasses import dataclass

@dataclass
class SearchResult:
    """Structured search result"""
    chunk_id: str
    score: float
    distance: float

class VectorIndex(ABC):
    """Abstract base class for all vector indexes"""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self._vectors: Dict[str, np.ndarray] = {}
        self._is_built = False

    @abstractmethod
    def build(self, vectors: Dict[str, np.ndarray]) -> None:
        """Build the index from vectors"""
        pass

    @abstractmethod
    def add(self, chunk_id: str, vector: np.ndarray) -> None:
        """Add a single vector to index"""
        pass

    @abstractmethod
    def remove(self, chunk_id: str) -> None:
        """Remove a vector from index"""
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int) -> List[SearchResult]:
        """Search for k nearest neighbors"""
        pass

    @abstractmethod
    def get_space_complexity(self) -> str:
        """Return space complexity as string"""
        pass

    @abstractmethod
    def get_time_complexity(self) -> str:
        """Return time complexity for search"""
        pass

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return dot_product / (norm_v1 * norm_v2)

    @staticmethod
    def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute Euclidean distance"""
        return np.linalg.norm(v1 - v2)

    def normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length"""
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector
```

### Task 2.2: Brute Force Index Implementation
```python
# infrastructure/indexes/brute_force.py
import numpy as np
from typing import List, Dict
import heapq

class BruteForceIndex(VectorIndex):
    """
    Brute-force linear scan index.

    Time Complexity:
    - Build: O(n) where n = number of vectors
    - Search: O(n * d) where d = dimension
    - Add: O(1)
    - Remove: O(1)

    Space Complexity: O(n * d)

    Rationale:
    - Guarantees exact results (no approximation)
    - Simple implementation with no preprocessing
    - Suitable for small datasets (<10k vectors)
    - Baseline for comparing other indexes
    """

    def __init__(self, dimension: int, normalize: bool = True):
        super().__init__(dimension)
        self.normalize = normalize
        self._vector_matrix: Optional[np.ndarray] = None
        self._id_to_index: Dict[str, int] = {}
        self._index_to_id: Dict[int, str] = {}

    def build(self, vectors: Dict[str, np.ndarray]) -> None:
        """Build index by storing all vectors in a matrix"""
        if not vectors:
            self._is_built = True
            return

        self._vectors = vectors
        chunk_ids = list(vectors.keys())
        vector_list = []

        for idx, chunk_id in enumerate(chunk_ids):
            vector = vectors[chunk_id]
            if self.normalize:
                vector = self.normalize_vector(vector)
            vector_list.append(vector)
            self._id_to_index[chunk_id] = idx
            self._index_to_id[idx] = chunk_id

        self._vector_matrix = np.vstack(vector_list)
        self._is_built = True

    def add(self, chunk_id: str, vector: np.ndarray) -> None:
        """Add vector with incremental update"""
        if chunk_id in self._vectors:
            raise DuplicateResourceError("Chunk", chunk_id)

        if self.normalize:
            vector = self.normalize_vector(vector)

        self._vectors[chunk_id] = vector

        # Update matrix
        if self._vector_matrix is None:
            self._vector_matrix = vector.reshape(1, -1)
            idx = 0
        else:
            idx = len(self._id_to_index)
            self._vector_matrix = np.vstack([self._vector_matrix, vector])

        self._id_to_index[chunk_id] = idx
        self._index_to_id[idx] = chunk_id

    def remove(self, chunk_id: str) -> None:
        """Remove vector and rebuild matrix"""
        if chunk_id not in self._vectors:
            raise ChunkNotFoundError(chunk_id)

        # Remove from storage
        del self._vectors[chunk_id]

        # Rebuild index mappings and matrix
        self.build(self._vectors)

    def search(self, query_vector: np.ndarray, k: int) -> List[SearchResult]:
        """
        Brute-force k-NN search using cosine similarity.

        Algorithm:
        1. Normalize query vector if enabled
        2. Compute cosine similarity with all vectors (vectorized)
        3. Use heap to find top-k
        """
        if not self._is_built or self._vector_matrix is None:
            return []

        if self.normalize:
            query_vector = self.normalize_vector(query_vector)

        # Vectorized cosine similarity (fast with NumPy)
        if self.normalize:
            # If normalized, cosine similarity = dot product
            similarities = np.dot(self._vector_matrix, query_vector)
        else:
            # Compute full cosine similarity
            similarities = np.array([
                self.cosine_similarity(query_vector, vec)
                for vec in self._vector_matrix
            ])

        # Get top-k using partial sort (more efficient than full sort)
        top_k_indices = np.argpartition(similarities, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])][::-1]

        results = []
        for idx in top_k_indices:
            chunk_id = self._index_to_id[idx]
            score = float(similarities[idx])
            distance = 1.0 - score  # Convert similarity to distance
            results.append(SearchResult(
                chunk_id=chunk_id,
                score=score,
                distance=distance
            ))

        return results

    def get_space_complexity(self) -> str:
        return "O(n * d) where n=vectors, d=dimension"

    def get_time_complexity(self) -> str:
        return "Search: O(n * d), Add: O(1), Remove: O(n)"
```

### Task 2.3: K-D Tree Index Implementation
```python
# infrastructure/indexes/kdtree.py
from typing import Optional, List, Tuple
import numpy as np
from dataclasses import dataclass
import heapq

@dataclass
class KDNode:
    """Node in K-D tree"""
    point: np.ndarray
    chunk_id: str
    left: Optional['KDNode'] = None
    right: Optional['KDNode'] = None
    axis: int = 0

class KDTreeIndex(VectorIndex):
    """
    K-D Tree for exact nearest neighbor search.

    Time Complexity:
    - Build: O(n log n) where n = number of vectors
    - Search: O(log n) average case for low dimensions
             O(n) worst case for high dimensions (curse of dimensionality)
    - Add: O(log n) average (rebalancing not implemented for simplicity)
    - Remove: O(log n) + rebuild cost

    Space Complexity: O(n * d)

    Rationale:
    - Exact search with better than linear performance in low-to-moderate dimensions
    - Demonstrates classic spatial partitioning algorithm
    - Educational value for understanding dimensionality curse
    - Works well for dimensions < 20, degrades for higher dimensions

    Limitations:
    - High-dimensional embeddings (768+) will degrade to O(n) search
    - No dynamic rebalancing (would require red-black tree logic)
    - Included to show algorithmic understanding
    """

    def __init__(self, dimension: int):
        super().__init__(dimension)
        self.root: Optional[KDNode] = None

    def build(self, vectors: Dict[str, np.ndarray]) -> None:
        """Build K-D tree from vectors"""
        if not vectors:
            self._is_built = True
            return

        self._vectors = vectors
        points = [(chunk_id, vec) for chunk_id, vec in vectors.items()]
        self.root = self._build_tree(points, depth=0)
        self._is_built = True

    def _build_tree(self, points: List[Tuple[str, np.ndarray]], depth: int) -> Optional[KDNode]:
        """Recursively build K-D tree"""
        if not points:
            return None

        # Select axis based on depth (cycle through dimensions)
        axis = depth % self.dimension

        # Sort points by axis and find median
        points.sort(key=lambda x: x[1][axis])
        median_idx = len(points) // 2

        chunk_id, point = points[median_idx]

        # Create node and recursively build subtrees
        node = KDNode(
            point=point,
            chunk_id=chunk_id,
            axis=axis,
            left=self._build_tree(points[:median_idx], depth + 1),
            right=self._build_tree(points[median_idx + 1:], depth + 1)
        )

        return node

    def add(self, chunk_id: str, vector: np.ndarray) -> None:
        """Add vector to tree"""
        if chunk_id in self._vectors:
            raise DuplicateResourceError("Chunk", chunk_id)

        self._vectors[chunk_id] = vector
        self.root = self._insert(self.root, chunk_id, vector, depth=0)

    def _insert(self, node: Optional[KDNode], chunk_id: str, point: np.ndarray, depth: int) -> KDNode:
        """Recursively insert node"""
        if node is None:
            return KDNode(point=point, chunk_id=chunk_id, axis=depth % self.dimension)

        axis = node.axis

        if point[axis] < node.point[axis]:
            node.left = self._insert(node.left, chunk_id, point, depth + 1)
        else:
            node.right = self._insert(node.right, chunk_id, point, depth + 1)

        return node

    def remove(self, chunk_id: str) -> None:
        """Remove vector and rebuild tree (simple approach)"""
        if chunk_id not in self._vectors:
            raise ChunkNotFoundError(chunk_id)

        del self._vectors[chunk_id]
        self.build(self._vectors)

    def search(self, query_vector: np.ndarray, k: int) -> List[SearchResult]:
        """
        K-NN search using K-D tree.

        Algorithm:
        1. Traverse tree to find initial leaf
        2. Backtrack and check if other branches could have closer points
        3. Use max-heap to maintain k best points
        """
        if not self._is_built or self.root is None:
            return []

        # Max heap to store k nearest (negative distance for max-heap)
        best = []  # List of (-distance, chunk_id)

        self._search_tree(self.root, query_vector, k, best)

        # Convert heap to sorted results
        results = []
        for neg_dist, chunk_id in sorted(best):
            distance = -neg_dist
            score = 1.0 / (1.0 + distance)  # Convert distance to similarity
            results.append(SearchResult(
                chunk_id=chunk_id,
                score=score,
                distance=distance
            ))

        return results[::-1]  # Reverse to get highest score first

    def _search_tree(self, node: Optional[KDNode], query: np.ndarray, k: int, best: List) -> None:
        """Recursive K-NN search with backtracking"""
        if node is None:
            return

        # Compute distance to current node
        dist = self.euclidean_distance(query, node.point)

        # Update best if needed
        if len(best) < k:
            heapq.heappush(best, (-dist, node.chunk_id))
        elif dist < -best[0][0]:  # Better than worst in heap
            heapq.heapreplace(best, (-dist, node.chunk_id))

        # Determine which subtree to search first
        axis = node.axis
        if query[axis] < node.point[axis]:
            near, far = node.left, node.right
        else:
            near, far = node.right, node.left

        # Search near subtree
        self._search_tree(near, query, k, best)

        # Check if we need to search far subtree
        # (if the hyperplane could intersect with sphere of k-th nearest)
        if len(best) < k or abs(query[axis] - node.point[axis]) < -best[0][0]:
            self._search_tree(far, query, k, best)

    def get_space_complexity(self) -> str:
        return "O(n * d) where n=vectors, d=dimension"

    def get_time_complexity(self) -> str:
        return "Build: O(n log n), Search: O(log n) avg (low dim), O(n) worst (high dim)"
```

### Task 2.4: LSH (Locality-Sensitive Hashing) Index
```python
# infrastructure/indexes/lsh.py
import numpy as np
from typing import List, Dict, Set
from collections import defaultdict
import hashlib

class LSHIndex(VectorIndex):
    """
    Locality-Sensitive Hashing for approximate nearest neighbor search.

    Uses random hyperplane hashing (suitable for cosine similarity).

    Time Complexity:
    - Build: O(n * L * h) where L=tables, h=hash bits per table
    - Search: O(L * b + c * d) where b=bucket size, c=candidates
    - Add: O(L * h)
    - Remove: O(L)

    Space Complexity: O(n * L) for hash tables + O(n * d) for vectors

    Rationale:
    - Sub-linear query time for high-dimensional data
    - Trades accuracy for speed (approximate results)
    - Configurable precision/recall via num_tables and hash_size
    - Demonstrates probabilistic data structure
    - Effective for dimensions > 100

    Parameters:
    - num_tables: More tables = higher recall but more memory
    - hash_size: More bits = finer buckets but more memory
    """

    def __init__(self, dimension: int, num_tables: int = 10, hash_size: int = 8):
        super().__init__(dimension)
        self.num_tables = num_tables
        self.hash_size = hash_size

        # Generate random hyperplanes for each table
        self.hyperplanes = [
            np.random.randn(hash_size, dimension)
            for _ in range(num_tables)
        ]

        # Hash tables: table_idx -> {hash -> set of chunk_ids}
        self.tables: List[Dict[str, Set[str]]] = [
            defaultdict(set) for _ in range(num_tables)
        ]

    def _hash_vector(self, vector: np.ndarray, table_idx: int) -> str:
        """
        Hash vector using random hyperplanes.

        Algorithm:
        1. Project vector onto random hyperplanes
        2. Create binary hash based on sign of projections
        3. Convert to string for dictionary key
        """
        projections = np.dot(self.hyperplanes[table_idx], vector)
        # Binary hash: 1 if positive, 0 if negative
        binary_hash = (projections > 0).astype(int)
        # Convert to string
        return ''.join(map(str, binary_hash))

    def build(self, vectors: Dict[str, np.ndarray]) -> None:
        """Build LSH hash tables"""
        if not vectors:
            self._is_built = True
            return

        self._vectors = vectors

        # Insert each vector into all hash tables
        for chunk_id, vector in vectors.items():
            for table_idx in range(self.num_tables):
                hash_key = self._hash_vector(vector, table_idx)
                self.tables[table_idx][hash_key].add(chunk_id)

        self._is_built = True

    def add(self, chunk_id: str, vector: np.ndarray) -> None:
        """Add vector to all hash tables"""
        if chunk_id in self._vectors:
            raise DuplicateResourceError("Chunk", chunk_id)

        self._vectors[chunk_id] = vector

        for table_idx in range(self.num_tables):
            hash_key = self._hash_vector(vector, table_idx)
            self.tables[table_idx][hash_key].add(chunk_id)

    def remove(self, chunk_id: str) -> None:
        """Remove vector from all hash tables"""
        if chunk_id not in self._vectors:
            raise ChunkNotFoundError(chunk_id)

        vector = self._vectors[chunk_id]

        # Remove from all tables
        for table_idx in range(self.num_tables):
            hash_key = self._hash_vector(vector, table_idx)
            self.tables[table_idx][hash_key].discard(chunk_id)

        del self._vectors[chunk_id]

    def search(self, query_vector: np.ndarray, k: int) -> List[SearchResult]:
        """
        Approximate k-NN search using LSH.

        Algorithm:
        1. Hash query into all tables
        2. Collect candidate chunk IDs from matching buckets
        3. Compute exact similarity for all candidates
        4. Return top-k
        """
        if not self._is_built:
            return []

        # Collect candidates from all tables
        candidates = set()
        for table_idx in range(self.num_tables):
            hash_key = self._hash_vector(query_vector, table_idx)
            candidates.update(self.tables[table_idx].get(hash_key, set()))

        if not candidates:
            # Fallback: if no candidates, check a few random buckets
            # (This improves recall at cost of some speed)
            for table_idx in range(min(3, self.num_tables)):
                for bucket in list(self.tables[table_idx].values())[:5]:
                    candidates.update(bucket)

        # Compute exact similarities for candidates
        scored_candidates = []
        for chunk_id in candidates:
            if chunk_id not in self._vectors:
                continue
            vector = self._vectors[chunk_id]
            similarity = self.cosine_similarity(query_vector, vector)
            distance = 1.0 - similarity
            scored_candidates.append((similarity, chunk_id, distance))

        # Sort by similarity (descending) and return top-k
        scored_candidates.sort(reverse=True, key=lambda x: x[0])

        results = []
        for similarity, chunk_id, distance in scored_candidates[:k]:
            results.append(SearchResult(
                chunk_id=chunk_id,
                score=similarity,
                distance=distance
            ))

        return results

    def get_space_complexity(self) -> str:
        return f"O(n * L) where L={self.num_tables} hash tables"

    def get_time_complexity(self) -> str:
        return "Search: O(L * b + c * d) sublinear on average, where b=bucket_size, c=candidates"
```

### Task 2.5: Index Factory & Manager
```python
# infrastructure/indexes/factory.py
from typing import Dict, Type
from enum import Enum

class IndexType(str, Enum):
    BRUTE_FORCE = "brute_force"
    KD_TREE = "kd_tree"
    LSH = "lsh"

class IndexFactory:
    """Factory for creating vector indexes"""

    _index_classes: Dict[IndexType, Type[VectorIndex]] = {
        IndexType.BRUTE_FORCE: BruteForceIndex,
        IndexType.KD_TREE: KDTreeIndex,
        IndexType.LSH: LSHIndex,
    }

    @classmethod
    def create_index(
        cls,
        index_type: IndexType,
        dimension: int,
        **kwargs
    ) -> VectorIndex:
        """
        Create index instance.

        Args:
            index_type: Type of index to create
            dimension: Vector dimension
            **kwargs: Index-specific parameters
                - LSH: num_tables, hash_size
                - BruteForce: normalize
        """
        if index_type not in cls._index_classes:
            raise ValueError(f"Unknown index type: {index_type}")

        index_class = cls._index_classes[index_type]
        return index_class(dimension, **kwargs)

    @classmethod
    def get_recommended_index(cls, num_vectors: int, dimension: int) -> IndexType:
        """Recommend index based on data characteristics"""
        if num_vectors < 1000:
            return IndexType.BRUTE_FORCE
        elif dimension < 20:
            return IndexType.KD_TREE
        else:
            return IndexType.LSH
```

---

## Phase 3: Concurrency & Thread Safety (Priority: CRITICAL)

### Task 3.1: Advanced Read-Write Lock Implementation
```python
# infrastructure/concurrency/locks.py
import threading
from contextlib import contextmanager
from typing import Optional
import logging

class ReadWriteLock:
    """
    Reader-Writer lock with writer priority.

    Design:
    - Multiple readers can hold lock simultaneously
    - Only one writer can hold lock (exclusive)
    - Writers have priority over readers (prevent writer starvation)

    Implementation:
    - Use threading.Condition for signaling
    - Track active readers and writers
    - Queue waiting writers
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._readers = 0
        self._writers = 0
        self._waiting_writers = 0
        self.logger = logging.getLogger(__name__)

    @contextmanager
    def read_lock(self):
        """Context manager for read lock"""
        self.acquire_read()
        try:
            yield
        finally:
            self.release_read()

    @contextmanager
    def write_lock(self):
        """Context manager for write lock"""
        self.acquire_write()
        try:
            yield
        finally:
            self.release_write()

    def acquire_read(self):
        """Acquire read lock (shared)"""
        with self._condition:
            # Wait if there are writers (give priority to writers)
            while self._writers > 0 or self._waiting_writers > 0:
                self._condition.wait()
            self._readers += 1
            self.logger.debug(f"Read lock acquired. Active readers: {self._readers}")

    def release_read(self):
        """Release read lock"""
        with self._condition:
            self._readers -= 1
            self.logger.debug(f"Read lock released. Active readers: {self._readers}")
            # Notify waiting writers if no readers
            if self._readers == 0:
                self._condition.notify_all()

    def acquire_write(self):
        """Acquire write lock (exclusive)"""
        with self._condition:
            self._waiting_writers += 1
            try:
                # Wait until no readers and no writers
                while self._readers > 0 or self._writers > 0:
                    self._condition.wait()
                self._writers = 1
                self.logger.debug("Write lock acquired")
            finally:
                self._waiting_writers -= 1

    def release_write(self):
        """Release write lock"""
        with self._condition:
            self._writers = 0
            self.logger.debug("Write lock released")
            # Notify all waiting threads
            self._condition.notify_all()

class AsyncReadWriteLock:
    """
    Asyncio-compatible read-write lock for async endpoints.

    Uses asyncio.Lock and asyncio.Condition for async/await support.
    """

    def __init__(self):
        import asyncio
        self._lock = asyncio.Lock()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0
        self._read_condition = asyncio.Condition(self._lock)
        self._write_condition = asyncio.Condition(self._lock)

    async def acquire_read(self):
        async with self._read_condition:
            while self._writer or self._waiting_writers > 0:
                await self._read_condition.wait()
            self._readers += 1

    async def release_read(self):
        async with self._read_condition:
            self._readers -= 1
            if self._readers == 0:
                self._read_condition.notify_all()

    async def acquire_write(self):
        async with self._write_condition:
            self._waiting_writers += 1
            try:
                while self._readers > 0 or self._writer:
                    await self._write_condition.wait()
                self._writer = True
            finally:
                self._waiting_writers -= 1

    async def release_write(self):
        async with self._write_condition:
            self._writer = False
            self._write_condition.notify_all()
```

### Task 3.2: Thread-Safe Repository Implementation
```python
# infrastructure/repositories/library_repository.py
from typing import Dict, Optional, List
from uuid import UUID
import threading

class InMemoryLibraryRepository:
    """
    Thread-safe in-memory repository for libraries.

    All operations are protected by read-write lock.
    """

    def __init__(self):
        self._libraries: Dict[str, Library] = {}
        self._lock = ReadWriteLock()

    def save(self, library: Library) -> Library:
        """Save or update library"""
        with self._lock.write_lock():
            self._libraries[str(library.id)] = library
            return library

    def find_by_id(self, library_id: str) -> Optional[Library]:
        """Find library by ID"""
        with self._lock.read_lock():
            return self._libraries.get(library_id)

    def find_all(self) -> List[Library]:
        """Get all libraries"""
        with self._lock.read_lock():
            return list(self._libraries.values())

    def delete(self, library_id: str) -> bool:
        """Delete library"""
        with self._lock.write_lock():
            if library_id in self._libraries:
                del self._libraries[library_id]
                return True
            return False

    def exists(self, library_id: str) -> bool:
        """Check if library exists"""
        with self._lock.read_lock():
            return library_id in self._libraries
```

---

## Phase 4: Service Layer & Business Logic (Priority: HIGH)

### Task 4.1: Library Service with SOLID Principles
```python
# application/services/library_service.py
from typing import List, Optional
from uuid import UUID
import logging

class LibraryService:
    """
    Application service for library operations.

    Responsibilities:
    - Business logic for library CRUD
    - Coordinate between repository and index manager
    - Validation and error handling
    - Transaction-like consistency
    """

    def __init__(
        self,
        repository: LibraryRepository,
        index_manager: IndexManager,
        logger: Optional[logging.Logger] = None
    ):
        self.repository = repository
        self.index_manager = index_manager
        self.logger = logger or logging.getLogger(__name__)

    def create_library(
        self,
        name: str,
        metadata: Optional[LibraryMetadata] = None
    ) -> Library:
        """
        Create new library.

        Steps:
        1. Validate input
        2. Create Library entity
        3. Save to repository
        4. Initialize empty index
        5. Return created library
        """
        self.logger.info(f"Creating library: {name}")

        # Create library entity
        library = Library(
            name=name,
            metadata=metadata or LibraryMetadata()
        )

        # Save to repository (thread-safe)
        saved_library = self.repository.save(library)

        # Initialize index
        self.index_manager.initialize_index(
            library_id=str(library.id),
            dimension=library.metadata.embedding_dimension,
            index_type=IndexType(library.metadata.index_type)
        )

        self.logger.info(f"Library created: {library.id}")
        return saved_library

    def get_library(self, library_id: str) -> Library:
        """Get library by ID"""
        library = self.repository.find_by_id(library_id)
        if not library:
            raise LibraryNotFoundError(library_id)
        return library

    def update_library(
        self,
        library_id: str,
        name: Optional[str] = None,
        metadata: Optional[LibraryMetadata] = None
    ) -> Library:
        """Update library metadata"""
        library = self.get_library(library_id)

        if name:
            library.name = name
        if metadata:
            library.metadata = metadata

        return self.repository.save(library)

    def delete_library(self, library_id: str) -> None:
        """
        Delete library and all associated data.

        Steps:
        1. Verify library exists
        2. Delete index
        3. Delete from repository
        """
        self.logger.info(f"Deleting library: {library_id}")

        if not self.repository.exists(library_id):
            raise LibraryNotFoundError(library_id)

        # Delete index first (cleanup)
        self.index_manager.delete_index(library_id)

        # Delete from repository
        self.repository.delete(library_id)

        self.logger.info(f"Library deleted: {library_id}")

    def list_libraries(self) -> List[Library]:
        """List all libraries"""
        return self.repository.find_all()
```

### Task 4.2: Document Service
```python
# application/services/document_service.py
from typing import List, Optional
import numpy as np

class DocumentService:
    """
    Service for document operations within a library.

    Responsibilities:
    - Add/remove documents
    - Manage chunks
    - Update library index when documents change
    - Generate embeddings if needed
    """

    def __init__(
        self,
        library_service: LibraryService,
        index_manager: IndexManager,
        embedding_service: Optional['EmbeddingService'] = None
    ):
        self.library_service = library_service
        self.index_manager = index_manager
        self.embedding_service = embedding_service
        self.logger = logging.getLogger(__name__)

    def add_document(
        self,
        library_id: str,
        document: Document,
        auto_index: bool = True
    ) -> Document:
        """
        Add document to library.

        Steps:
        1. Get library (validates existence)
        2. Validate document embeddings
        3. Add document to library
        4. Update index with chunk embeddings
        5. Save library
        """
        self.logger.info(f"Adding document {document.id} to library {library_id}")

        library = self.library_service.get_library(library_id)

        # Validate embedding dimensions
        expected_dim = library.metadata.embedding_dimension
        for chunk in document.chunks:
            if len(chunk.embedding) != expected_dim:
                raise EmbeddingDimensionMismatchError(
                    expected=expected_dim,
                    got=len(chunk.embedding)
                )

        # Check for duplicate document ID
        if any(doc.id == document.id for doc in library.documents):
            raise DuplicateResourceError("Document", str(document.id))

        # Add to library
        library.documents.append(document)

        # Update index if auto-indexing enabled
        if auto_index:
            for chunk in document.chunks:
                self.index_manager.add_vector(
                    library_id=library_id,
                    chunk_id=str(chunk.id),
                    vector=np.array(chunk.embedding)
                )

        # Save updated library
        self.library_service.repository.save(library)

        self.logger.info(f"Document {document.id} added successfully")
        return document

    def get_document(self, library_id: str, document_id: str) -> Document:
        """Get document from library"""
        library = self.library_service.get_library(library_id)

        for doc in library.documents:
            if str(doc.id) == document_id:
                return doc

        raise DocumentNotFoundError(document_id)

    def delete_document(self, library_id: str, document_id: str) -> None:
        """
        Delete document from library.

        Steps:
        1. Find document
        2. Remove chunks from index
        3. Remove document from library
        4. Save library
        """
        self.logger.info(f"Deleting document {document_id} from library {library_id}")

        library = self.library_service.get_library(library_id)

        # Find document
        document = None
        for idx, doc in enumerate(library.documents):
            if str(doc.id) == document_id:
                document = doc
                library.documents.pop(idx)
                break

        if not document:
            raise DocumentNotFoundError(document_id)

        # Remove chunks from index
        for chunk in document.chunks:
            try:
                self.index_manager.remove_vector(library_id, str(chunk.id))
            except Exception as e:
                self.logger.warning(f"Failed to remove chunk {chunk.id} from index: {e}")

        # Save updated library
        self.library_service.repository.save(library)

        self.logger.info(f"Document {document_id} deleted successfully")

    def update_document(
        self,
        library_id: str,
        document_id: str,
        metadata: Optional[DocumentMetadata] = None,
        chunks: Optional[List[Chunk]] = None
    ) -> Document:
        """Update document metadata or chunks"""
        # Delete old version and add new one (simple approach)
        old_doc = self.get_document(library_id, document_id)

        updated_doc = Document(
            id=old_doc.id,
            chunks=chunks if chunks is not None else old_doc.chunks,
            metadata=metadata if metadata is not None else old_doc.metadata
        )

        self.delete_document(library_id, document_id)
        return self.add_document(library_id, updated_doc)
```

### Task 4.3: Query Service with Metadata Filtering
```python
# application/services/query_service.py
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import re

@dataclass
class MetadataFilter:
    """Metadata filter specification"""
    field: str
    operator: str  # eq, ne, gt, gte, lt, lte, contains, in
    value: Any

@dataclass
class QueryRequest:
    """Query request with all parameters"""
    library_id: str
    query_vector: Optional[np.ndarray] = None
    query_text: Optional[str] = None
    k: int = 10
    filters: Optional[List[MetadataFilter]] = None
    min_score: float = 0.0

@dataclass
class QueryResponse:
    """Query response with metadata"""
    results: List[SearchResult]
    query_time_ms: float
    total_candidates: int
    filters_applied: bool

class QueryService:
    """
    Service for querying vector database.

    Features:
    - Vector similarity search
    - Metadata filtering (pre-filtering and post-filtering)
    - Text-to-embedding conversion
    - Result ranking and filtering
    """

    def __init__(
        self,
        library_service: LibraryService,
        index_manager: IndexManager,
        embedding_service: Optional['EmbeddingService'] = None
    ):
        self.library_service = library_service
        self.index_manager = index_manager
        self.embedding_service = embedding_service
        self.logger = logging.getLogger(__name__)

    def query(self, request: QueryRequest) -> QueryResponse:
        """
        Execute vector search query with optional filters.

        Algorithm:
        1. Get query vector (from vector or text)
        2. Apply pre-filtering if filters present
        3. Execute vector search
        4. Apply post-filtering if needed
        5. Return top-k results
        """
        import time
        start_time = time.time()

        # Get library
        library = self.library_service.get_library(request.library_id)

        # Get query vector
        if request.query_vector is not None:
            query_vector = request.query_vector
        elif request.query_text is not None:
            if not self.embedding_service:
                raise InvalidOperationError("EmbeddingService required for text queries")
            query_vector = self.embedding_service.embed(request.query_text)
        else:
            raise InvalidOperationError("Either query_vector or query_text must be provided")

        # Apply pre-filtering to get candidate chunks
        candidate_chunks = self._get_candidate_chunks(library, request.filters)

        if request.filters and candidate_chunks is not None:
            # Search only within filtered candidates
            results = self._search_with_candidates(
                request.library_id,
                query_vector,
                request.k,
                candidate_chunks
            )
        else:
            # Standard index search
            results = self.index_manager.search(
                library_id=request.library_id,
                query_vector=query_vector,
                k=request.k
            )

        # Apply minimum score filter
        results = [r for r in results if r.score >= request.min_score]

        # Enrich results with chunk data
        enriched_results = self._enrich_results(library, results)

        query_time = (time.time() - start_time) * 1000  # Convert to ms

        return QueryResponse(
            results=enriched_results,
            query_time_ms=query_time,
            total_candidates=len(candidate_chunks) if candidate_chunks else -1,
            filters_applied=request.filters is not None
        )

    def _get_candidate_chunks(
        self,
        library: Library,
        filters: Optional[List[MetadataFilter]]
    ) -> Optional[Set[str]]:
        """
        Pre-filter chunks by metadata.

        Returns set of chunk IDs that match filters, or None if no filters.
        """
        if not filters:
            return None

        candidate_ids = set()

        # Iterate through all chunks in library
        for document in library.documents:
            for chunk in document.chunks:
                if self._matches_filters(chunk.metadata, filters):
                    candidate_ids.add(str(chunk.id))

        return candidate_ids

    def _matches_filters(
        self,
        metadata: ChunkMetadata,
        filters: List[MetadataFilter]
    ) -> bool:
        """Check if metadata matches all filters (AND logic)"""
        for filter in filters:
            if not self._matches_single_filter(metadata, filter):
                return False
        return True

    def _matches_single_filter(
        self,
        metadata: ChunkMetadata,
        filter: MetadataFilter
    ) -> bool:
        """Check if metadata matches a single filter"""
        # Get value from metadata
        if hasattr(metadata, filter.field):
            value = getattr(metadata, filter.field)
        elif filter.field in metadata.custom_fields:
            value = metadata.custom_fields[filter.field]
        else:
            return False  # Field doesn't exist

        # Apply operator
        if filter.operator == "eq":
            return value == filter.value
        elif filter.operator == "ne":
            return value != filter.value
        elif filter.operator == "gt":
            return value > filter.value
        elif filter.operator == "gte":
            return value >= filter.value
        elif filter.operator == "lt":
            return value < filter.value
        elif filter.operator == "lte":
            return value <= filter.value
        elif filter.operator == "contains":
            return str(filter.value) in str(value)
        elif filter.operator == "in":
            return value in filter.value
        else:
            raise ValueError(f"Unknown operator: {filter.operator}")

    def _search_with_candidates(
        self,
        library_id: str,
        query_vector: np.ndarray,
        k: int,
        candidate_ids: Set[str]
    ) -> List[SearchResult]:
        """
        Search within pre-filtered candidates.

        Since index may not support filtering, we:
        1. Get more results from index (e.g., k*10)
        2. Filter to candidates
        3. Return top-k
        """
        # Get more results to account for filtering
        expanded_k = min(k * 10, 1000)
        all_results = self.index_manager.search(
            library_id=library_id,
            query_vector=query_vector,
            k=expanded_k
        )

        # Filter to candidates
        filtered_results = [
            r for r in all_results
            if r.chunk_id in candidate_ids
        ]

        # Return top-k
        return filtered_results[:k]

    def _enrich_results(
        self,
        library: Library,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """Add chunk text and metadata to results"""
        # Build chunk lookup
        chunk_map = {}
        for doc in library.documents:
            for chunk in doc.chunks:
                chunk_map[str(chunk.id)] = (chunk, doc)

        # Enrich each result
        enriched = []
        for result in results:
            if result.chunk_id in chunk_map:
                chunk, doc = chunk_map[result.chunk_id]
                # You could create an enriched result object here
                # For now, just pass through
                enriched.append(result)

        return enriched
```

### Task 4.4: Embedding Service (Cohere Integration)
```python
# application/services/embedding_service.py
import os
from typing import List
import numpy as np
import requests

class EmbeddingService:
    """
    Service for generating embeddings using Cohere API.

    Supports:
    - Single text embedding
    - Batch embedding
    - Caching
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "embed-english-v3.0"):
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("Cohere API key required")
        self.model = model
        self.base_url = "https://api.cohere.ai/v1"
        self.logger = logging.getLogger(__name__)

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for single text"""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for batch of texts.

        Uses Cohere's embed endpoint.
        """
        self.logger.info(f"Generating embeddings for {len(texts)} texts")

        url = f"{self.base_url}/embed"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "texts": texts,
            "model": self.model,
            "input_type": "search_document"  # or "search_query" for queries
        }

        response = requests.post(url, json=payload, headers=headers)

        if response.status_code != 200:
            raise Exception(f"Cohere API error: {response.text}")

        data = response.json()
        embeddings = [np.array(emb) for emb in data["embeddings"]]

        self.logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings
```

---

## Phase 5: FastAPI Layer (Priority: HIGH)

### Task 5.1: API Router Structure
```python
# api/routes/libraries.py
from fastapi import APIRouter, Depends, status, HTTPException
from typing import List
from uuid import UUID

router = APIRouter(prefix="/libraries", tags=["libraries"])

# Dependency injection
def get_library_service() -> LibraryService:
    # Returns singleton service instance
    # In production, use proper DI container
    from api.dependencies import get_service_container
    return get_service_container().library_service

@router.post(
    "",
    response_model=LibraryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create new library",
    description="Create a new library for storing documents and embeddings"
)
async def create_library(
    request: LibraryCreateRequest,
    service: LibraryService = Depends(get_library_service)
) -> LibraryResponse:
    """Create new library"""
    try:
        library = service.create_library(
            name=request.name,
            metadata=request.metadata
        )
        return LibraryResponse.from_domain(library)
    except VectorDBException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.exception("Unexpected error creating library")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.get(
    "",
    response_model=List[LibraryResponse],
    summary="List all libraries"
)
async def list_libraries(
    service: LibraryService = Depends(get_library_service)
) -> List[LibraryResponse]:
    """List all libraries"""
    libraries = service.list_libraries()
    return [LibraryResponse.from_domain(lib) for lib in libraries]

@router.get(
    "/{library_id}",
    response_model=LibraryDetailResponse,
    summary="Get library by ID"
)
async def get_library(
    library_id: UUID,
    service: LibraryService = Depends(get_library_service)
) -> LibraryDetailResponse:
    """Get library details"""
    try:
        library = service.get_library(str(library_id))
        return LibraryDetailResponse.from_domain(library)
    except LibraryNotFoundError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)

@router.delete(
    "/{library_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete library"
)
async def delete_library(
    library_id: UUID,
    service: LibraryService = Depends(get_library_service)
) -> None:
    """Delete library and all its data"""
    try:
        service.delete_library(str(library_id))
    except LibraryNotFoundError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)

# Document routes
@router.post(
    "/{library_id}/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add document to library"
)
async def add_document(
    library_id: UUID,
    request: DocumentCreateRequest,
    service: DocumentService = Depends(get_document_service)
) -> DocumentResponse:
    """Add new document with chunks"""
    try:
        document = request.to_domain()
        added_doc = service.add_document(str(library_id), document)
        return DocumentResponse.from_domain(added_doc)
    except VectorDBException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)

# Query routes
@router.post(
    "/{library_id}/query",
    response_model=QueryResponseSchema,
    summary="Query library for similar chunks"
)
async def query_library(
    library_id: UUID,
    request: QueryRequestSchema,
    service: QueryService = Depends(get_query_service)
) -> QueryResponseSchema:
    """Execute similarity search"""
    try:
        query_req = QueryRequest(
            library_id=str(library_id),
            query_vector=request.embedding,
            query_text=request.text,
            k=request.k,
            filters=request.filters,
            min_score=request.min_score
        )
        response = service.query(query_req)
        return QueryResponseSchema.from_domain(response)
    except VectorDBException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
```

### Task 5.2: API Schemas (DTOs)
```python
# api/schemas/library.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime

class LibraryMetadataSchema(BaseModel):
    """API schema for library metadata"""
    description: Optional[str] = None
    index_type: str = "brute_force"
    embedding_dimension: int = Field(768, ge=1, le=4096)

class LibraryCreateRequest(BaseModel):
    """Request to create library"""
    name: str = Field(..., min_length=1, max_length=255)
    metadata: Optional[LibraryMetadataSchema] = None

    class Config:
        schema_extra = {
            "example": {
                "name": "Research Papers",
                "metadata": {
                    "description": "Collection of AI research papers",
                    "index_type": "lsh",
                    "embedding_dimension": 768
                }
            }
        }

class LibraryResponse(BaseModel):
    """Response with library info"""
    id: UUID
    name: str
    metadata: LibraryMetadataSchema
    document_count: int
    created_at: datetime

    @classmethod
    def from_domain(cls, library: Library) -> 'LibraryResponse':
        return cls(
            id=library.id,
            name=library.name,
            metadata=LibraryMetadataSchema(**library.metadata.dict()),
            document_count=len(library.documents),
            created_at=library.metadata.created_at
        )

# Similar schemas for Document, Chunk, Query, etc.
```

### Task 5.3: Global Exception Handler
```python
# api/middleware/exception_handler.py
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

async def vector_db_exception_handler(request: Request, exc: VectorDBException):
    """Handle domain exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": exc.__class__.__name__,
                "message": exc.message,
                "path": request.url.path
            }
        }
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "type": "ValidationError",
                "message": "Request validation failed",
                "details": exc.errors()
            }
        }
    )

async def generic_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler"""
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "type": "InternalServerError",
                "message": "An unexpected error occurred"
            }
        }
    )

# Register in main app
app.add_exception_handler(VectorDBException, vector_db_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)
```

---

## Phase 6: Persistence Layer (Priority: MEDIUM)

### Task 6.1: Persistence Strategy Design
**Hybrid Approach:**
1. **Write-Ahead Log (WAL)**: Append-only log for all mutations
2. **Periodic Snapshots**: Full state snapshots every N minutes
3. **Index Serialization**: Separate index file per library

**Trade-offs:**
- WAL ensures durability (no data loss)
- Snapshots enable fast recovery
- Separate index files allow selective loading

### Task 6.2: WAL Implementation
```python
# infrastructure/persistence/wal.py
import os
import json
from typing import Dict, Any
from datetime import datetime
from enum import Enum

class OperationType(str, Enum):
    CREATE_LIBRARY = "create_library"
    DELETE_LIBRARY = "delete_library"
    ADD_DOCUMENT = "add_document"
    DELETE_DOCUMENT = "delete_document"
    UPDATE_METADATA = "update_metadata"

@dataclass
class WALEntry:
    """Write-ahead log entry"""
    timestamp: datetime
    operation: OperationType
    library_id: str
    data: Dict[str, Any]
    sequence_number: int

class WriteAheadLog:
    """
    Write-ahead log for durability.

    Design:
    - Append-only file
    - Each line is a JSON operation
    - Fsync after each write for durability
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._file = None
        self._sequence = 0
        self._open_log()

    def _open_log(self):
        """Open log file in append mode"""
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self._file = open(self.file_path, 'a')

        # Read existing entries to get sequence number
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    self._sequence = max(self._sequence, entry.get('sequence', 0))

    def append(self, operation: OperationType, library_id: str, data: Dict[str, Any]) -> None:
        """Append operation to log"""
        self._sequence += 1
        entry = WALEntry(
            timestamp=datetime.utcnow(),
            operation=operation,
            library_id=library_id,
            data=data,
            sequence_number=self._sequence
        )

        # Write as JSON line
        line = json.dumps({
            'timestamp': entry.timestamp.isoformat(),
            'operation': entry.operation,
            'library_id': entry.library_id,
            'data': entry.data,
            'sequence': entry.sequence_number
        })

        self._file.write(line + '\n')
        self._file.flush()
        os.fsync(self._file.fileno())  # Ensure written to disk

    def replay(self) -> List[WALEntry]:
        """Read all entries for recovery"""
        entries = []
        with open(self.file_path, 'r') as f:
            for line in f:
                entry_dict = json.loads(line)
                entries.append(WALEntry(
                    timestamp=datetime.fromisoformat(entry_dict['timestamp']),
                    operation=OperationType(entry_dict['operation']),
                    library_id=entry_dict['library_id'],
                    data=entry_dict['data'],
                    sequence_number=entry_dict['sequence']
                ))
        return entries

    def truncate(self) -> None:
        """Clear log (after snapshot)"""
        self._file.close()
        self._file = open(self.file_path, 'w')
        self._sequence = 0
```

### Task 6.3: Snapshot Manager
```python
# infrastructure/persistence/snapshot.py
import pickle
import gzip
import numpy as np
from pathlib import Path

class SnapshotManager:
    """
    Manage periodic snapshots of database state.

    Format:
    - Libraries: JSON metadata
    - Vectors: NumPy .npz format (compressed)
    - Indexes: Pickled (with version compatibility)
    """

    def __init__(self, snapshot_dir: str):
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def save_snapshot(
        self,
        libraries: Dict[str, Library],
        indexes: Dict[str, VectorIndex]
    ) -> str:
        """Save full snapshot"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        snapshot_path = self.snapshot_dir / f"snapshot_{timestamp}"
        snapshot_path.mkdir(exist_ok=True)

        # Save libraries metadata (without embeddings)
        libraries_file = snapshot_path / "libraries.json"
        libraries_data = {
            lib_id: self._library_to_dict(lib)
            for lib_id, lib in libraries.items()
        }
        with open(libraries_file, 'w') as f:
            json.dump(libraries_data, f, indent=2, default=str)

        # Save vectors separately (compressed)
        vectors_dir = snapshot_path / "vectors"
        vectors_dir.mkdir(exist_ok=True)
        for lib_id, library in libraries.items():
            vectors = {}
            for doc in library.documents:
                for chunk in doc.chunks:
                    vectors[str(chunk.id)] = np.array(chunk.embedding)

            if vectors:
                np.savez_compressed(
                    vectors_dir / f"{lib_id}.npz",
                    **{chunk_id: vec for chunk_id, vec in vectors.items()}
                )

        # Save indexes (pickled)
        indexes_dir = snapshot_path / "indexes"
        indexes_dir.mkdir(exist_ok=True)
        for lib_id, index in indexes.items():
            with gzip.open(indexes_dir / f"{lib_id}.pkl.gz", 'wb') as f:
                pickle.dump(index, f)

        return str(snapshot_path)

    def load_latest_snapshot(self) -> Optional[Tuple[Dict, Dict]]:
        """Load most recent snapshot"""
        snapshots = sorted(self.snapshot_dir.glob("snapshot_*"))
        if not snapshots:
            return None

        latest = snapshots[-1]

        # Load libraries
        with open(latest / "libraries.json", 'r') as f:
            libraries_data = json.load(f)
        libraries = {
            lib_id: self._dict_to_library(data)
            for lib_id, data in libraries_data.items()
        }

        # Load indexes
        indexes = {}
        indexes_dir = latest / "indexes"
        if indexes_dir.exists():
            for index_file in indexes_dir.glob("*.pkl.gz"):
                lib_id = index_file.stem.replace('.pkl', '')
                with gzip.open(index_file, 'rb') as f:
                    indexes[lib_id] = pickle.load(f)

        return libraries, indexes

    def _library_to_dict(self, library: Library) -> Dict:
        """Serialize library to dict (without embeddings)"""
        return library.dict(exclude={'documents': {'__all__': {'chunks': {'__all__': {'embedding'}}}}})

    def _dict_to_library(self, data: Dict) -> Library:
        """Deserialize library from dict"""
        return Library(**data)
```

---

## Phase 7: Testing Strategy (Priority: HIGH)

### Task 7.1: Unit Tests for Indexes
```python
# tests/unit/test_indexes.py
import pytest
import numpy as np
from infrastructure.indexes import BruteForceIndex, KDTreeIndex, LSHIndex

class TestBruteForceIndex:
    """Test brute-force index"""

    @pytest.fixture
    def index(self):
        return BruteForceIndex(dimension=128)

    @pytest.fixture
    def sample_vectors(self):
        """Generate sample vectors with known similarities"""
        np.random.seed(42)
        return {
            'chunk1': np.random.randn(128),
            'chunk2': np.random.randn(128),
            'chunk3': np.random.randn(128),
        }

    def test_build_index(self, index, sample_vectors):
        """Test index building"""
        index.build(sample_vectors)
        assert index._is_built
        assert len(index._vectors) == 3

    def test_search_returns_correct_k(self, index, sample_vectors):
        """Test that search returns k results"""
        index.build(sample_vectors)
        query = np.random.randn(128)
        results = index.search(query, k=2)
        assert len(results) == 2

    def test_search_ordering(self, index, sample_vectors):
        """Test that results are ordered by score"""
        index.build(sample_vectors)
        query = sample_vectors['chunk1']  # Query with exact match
        results = index.search(query, k=3)

        # First result should be exact match
        assert results[0].chunk_id == 'chunk1'
        assert results[0].score > 0.99  # Near 1.0 for exact match

        # Scores should be descending
        assert results[0].score >= results[1].score >= results[2].score

    def test_add_vector(self, index, sample_vectors):
        """Test adding vector after build"""
        index.build(sample_vectors)
        new_vector = np.random.randn(128)
        index.add('chunk4', new_vector)
        assert 'chunk4' in index._vectors

    def test_remove_vector(self, index, sample_vectors):
        """Test removing vector"""
        index.build(sample_vectors)
        index.remove('chunk1')
        assert 'chunk1' not in index._vectors

    def test_duplicate_add_raises_error(self, index, sample_vectors):
        """Test that adding duplicate raises error"""
        index.build(sample_vectors)
        with pytest.raises(DuplicateResourceError):
            index.add('chunk1', np.random.randn(128))

class TestKDTreeIndex:
    """Test K-D tree index"""

    def test_exact_search(self):
        """Test that KD-tree returns exact results"""
        # Use low dimension for KD-tree efficiency
        index = KDTreeIndex(dimension=3)
        vectors = {
            'a': np.array([0, 0, 0]),
            'b': np.array([1, 0, 0]),
            'c': np.array([0, 1, 0]),
        }
        index.build(vectors)

        query = np.array([0.1, 0, 0])  # Closest to 'a'
        results = index.search(query, k=1)
        assert results[0].chunk_id == 'a'

class TestLSHIndex:
    """Test LSH index"""

    def test_approximate_search(self):
        """Test that LSH returns reasonable results"""
        index = LSHIndex(dimension=128, num_tables=5, hash_size=8)

        # Generate clustered vectors
        np.random.seed(42)
        cluster1 = [np.random.randn(128) + np.array([5]*128) for _ in range(10)]
        cluster2 = [np.random.randn(128) - np.array([5]*128) for _ in range(10)]

        vectors = {}
        for i, vec in enumerate(cluster1 + cluster2):
            vectors[f'chunk{i}'] = vec

        index.build(vectors)

        # Query with cluster1 point should return cluster1 chunks
        query = cluster1[0]
        results = index.search(query, k=5)

        # At least some results should be from cluster1
        cluster1_chunks = set(f'chunk{i}' for i in range(10))
        result_chunks = set(r.chunk_id for r in results)
        assert len(result_chunks & cluster1_chunks) >= 3  # Reasonable recall
```

### Task 7.2: Integration Tests
```python
# tests/integration/test_api.py
import pytest
from fastapi.testclient import TestClient
from uuid import uuid4

@pytest.fixture
def client():
    """FastAPI test client"""
    from api.main import app
    return TestClient(app)

class TestLibraryAPI:
    """Integration tests for library endpoints"""

    def test_create_library(self, client):
        """Test library creation"""
        response = client.post("/libraries", json={
            "name": "Test Library",
            "metadata": {
                "description": "Test",
                "index_type": "brute_force"
            }
        })
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["name"] == "Test Library"

    def test_get_library(self, client):
        """Test retrieving library"""
        # Create library
        create_resp = client.post("/libraries", json={"name": "Test"})
        library_id = create_resp.json()["id"]

        # Get library
        get_resp = client.get(f"/libraries/{library_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["id"] == library_id

    def test_delete_library(self, client):
        """Test library deletion"""
        # Create
        create_resp = client.post("/libraries", json={"name": "Test"})
        library_id = create_resp.json()["id"]

        # Delete
        delete_resp = client.delete(f"/libraries/{library_id}")
        assert delete_resp.status_code == 204

        # Verify deleted
        get_resp = client.get(f"/libraries/{library_id}")
        assert get_resp.status_code == 404

class TestQueryAPI:
    """Integration tests for query endpoint"""

    def test_query_with_embedding(self, client):
        """Test querying with embedding vector"""
        # Setup: Create library and add document
        lib_resp = client.post("/libraries", json={"name": "Test"})
        lib_id = lib_resp.json()["id"]

        doc_resp = client.post(f"/libraries/{lib_id}/documents", json={
            "metadata": {"title": "Test Doc"},
            "chunks": [{
                "text": "Test chunk",
                "embedding": [0.1] * 768
            }]
        })

        # Query
        query_resp = client.post(f"/libraries/{lib_id}/query", json={
            "embedding": [0.1] * 768,
            "k": 1
        })

        assert query_resp.status_code == 200
        results = query_resp.json()["results"]
        assert len(results) == 1
```

### Task 7.3: Thread-Safety Tests
```python
# tests/concurrency/test_thread_safety.py
import pytest
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class TestConcurrency:
    """Test thread-safety of repository and indexes"""

    def test_concurrent_writes(self, library_service):
        """Test multiple concurrent writes"""
        def create_library(i):
            return library_service.create_library(f"Library {i}")

        # Create 100 libraries concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_library, i) for i in range(100)]
            results = [f.result() for f in as_completed(futures)]

        # All should succeed
        assert len(results) == 100
        assert len(set(r.id for r in results)) == 100  # All unique

    def test_concurrent_reads_and_writes(self, library_service, document_service):
        """Test concurrent reads during writes"""
        # Create library
        library = library_service.create_library("Test")

        results = []
        errors = []

        def write_document(i):
            try:
                doc = Document(
                    chunks=[Chunk(text=f"Chunk {i}", embedding=[0.1]*768)]
                )
                document_service.add_document(str(library.id), doc)
                results.append(i)
            except Exception as e:
                errors.append(e)

        def read_library():
            try:
                lib = library_service.get_library(str(library.id))
                results.append(len(lib.documents))
            except Exception as e:
                errors.append(e)

        # Run mixed workload
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for i in range(50):
                futures.append(executor.submit(write_document, i))
            for _ in range(50):
                futures.append(executor.submit(read_library))

            for f in as_completed(futures):
                f.result()

        # No errors should occur
        assert len(errors) == 0
```

---

## Phase 8: Docker & Deployment (Priority: MEDIUM)

### Task 8.1: Multi-Stage Dockerfile
```dockerfile
# Dockerfile
# Stage 1: Build dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY src/ ./src/
COPY api/ ./api/

# Create data directory
RUN mkdir -p /data/snapshots /data/wal

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/data
ENV PATH=/root/.local/bin:$PATH

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Task 8.2: Docker Compose (with Temporal)
```yaml
# docker-compose.yml
version: '3.8'

services:
  vectordb:
    build: .
    ports:
      - "8000:8000"
    environment:
      - COHERE_API_KEY=${COHERE_API_KEY}
      - TEMPORAL_HOST=temporal:7233
    volumes:
      - vectordb_data:/data
    depends_on:
      - temporal
    networks:
      - vectordb_network

  temporal:
    image: temporalio/auto-setup:latest
    ports:
      - "7233:7233"
      - "8233:8233"
    environment:
      - DB=postgresql
      - DB_PORT=5432
      - POSTGRES_USER=temporal
      - POSTGRES_PWD=temporal
      - POSTGRES_SEEDS=postgresql
    depends_on:
      - postgresql
    networks:
      - vectordb_network

  postgresql:
    image: postgres:13
    environment:
      - POSTGRES_PASSWORD=temporal
      - POSTGRES_USER=temporal
    volumes:
      - temporal_db:/var/lib/postgresql/data
    networks:
      - vectordb_network

  temporal-ui:
    image: temporalio/ui:latest
    ports:
      - "8080:8080"
    environment:
      - TEMPORAL_ADDRESS=temporal:7233
    depends_on:
      - temporal
    networks:
      - vectordb_network

volumes:
  vectordb_data:
  temporal_db:

networks:
  vectordb_network:
    driver: bridge
```

---

## Phase 9: Extra Features (Priority: LOW-MEDIUM)

### Task 9.1: Python SDK Client
```python
# sdk/vectordb_client.py
import requests
from typing import List, Optional, Dict, Any
import numpy as np

class VectorDBClient:
    """
    Python SDK for Vector Database API.

    Usage:
        client = VectorDBClient("http://localhost:8000")
        library = client.create_library("My Library")
        client.add_document(library.id, text="Document text", ...)
        results = client.query(library.id, query_text="search query", k=5)
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        if api_key:
            self.session.headers['Authorization'] = f'Bearer {api_key}'

    def create_library(
        self,
        name: str,
        description: Optional[str] = None,
        index_type: str = "brute_force"
    ) -> Dict[str, Any]:
        """Create new library"""
        response = self.session.post(
            f"{self.base_url}/libraries",
            json={
                "name": name,
                "metadata": {
                    "description": description,
                    "index_type": index_type
                }
            }
        )
        response.raise_for_status()
        return response.json()

    def add_document(
        self,
        library_id: str,
        text: str,
        metadata: Optional[Dict] = None,
        chunks: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Add document to library"""
        # If chunks not provided, create single chunk
        if chunks is None:
            chunks = [{"text": text, "embedding": None}]

        response = self.session.post(
            f"{self.base_url}/libraries/{library_id}/documents",
            json={
                "chunks": chunks,
                "metadata": metadata or {}
            }
        )
        response.raise_for_status()
        return response.json()

    def query(
        self,
        library_id: str,
        query_text: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        k: int = 10,
        filters: Optional[List[Dict]] = None,
        min_score: float = 0.0
    ) -> Dict[str, Any]:
        """Query library for similar chunks"""
        if query_text is None and query_embedding is None:
            raise ValueError("Either query_text or query_embedding must be provided")

        response = self.session.post(
            f"{self.base_url}/libraries/{library_id}/query",
            json={
                "text": query_text,
                "embedding": query_embedding,
                "k": k,
                "filters": filters,
                "min_score": min_score
            }
        )
        response.raise_for_status()
        return response.json()

    def delete_library(self, library_id: str) -> None:
        """Delete library"""
        response = self.session.delete(f"{self.base_url}/libraries/{library_id}")
        response.raise_for_status()
```

### Task 9.2: Temporal Workflows (Optional)
```python
# temporal/workflows/query_workflow.py
from temporalio import workflow, activity
from datetime import timedelta
from typing import List
import numpy as np

@activity.defn
async def preprocess_query(query_text: str) -> str:
    """Preprocess query text"""
    # Could do: lowercasing, tokenization, etc.
    return query_text.lower().strip()

@activity.defn
async def generate_embedding(text: str) -> List[float]:
    """Generate embedding using Cohere"""
    # Call embedding service
    from application.services import EmbeddingService
    service = EmbeddingService()
    embedding = service.embed(text)
    return embedding.tolist()

@activity.defn
async def retrieve_chunks(
    library_id: str,
    embedding: List[float],
    k: int
) -> List[Dict]:
    """Retrieve relevant chunks"""
    from application.services import QueryService
    service = QueryService(...)
    results = service.query(QueryRequest(
        library_id=library_id,
        query_vector=np.array(embedding),
        k=k
    ))
    return [r.dict() for r in results.results]

@activity.defn
async def rerank_results(results: List[Dict]) -> List[Dict]:
    """Rerank results using more expensive model"""
    # Could call a reranking model here
    return results

@workflow.defn
class QueryWorkflow:
    """
    Durable query workflow.

    Steps:
    1. Preprocess query
    2. Generate embedding
    3. Retrieve chunks
    4. Rerank (optional)
    5. Return results
    """

    @workflow.run
    async def run(self, library_id: str, query_text: str, k: int = 10) -> List[Dict]:
        # Step 1: Preprocess
        processed_text = await workflow.execute_activity(
            preprocess_query,
            query_text,
            start_to_close_timeout=timedelta(seconds=10)
        )

        # Step 2: Generate embedding
        embedding = await workflow.execute_activity(
            generate_embedding,
            processed_text,
            start_to_close_timeout=timedelta(seconds=30)
        )

        # Step 3: Retrieve
        results = await workflow.execute_activity(
            retrieve_chunks,
            args=[library_id, embedding, k],
            start_to_close_timeout=timedelta(seconds=60)
        )

        # Step 4: Rerank (optional)
        if len(results) > 0:
            results = await workflow.execute_activity(
                rerank_results,
                results,
                start_to_close_timeout=timedelta(seconds=30)
            )

        return results

# Worker setup
async def start_worker():
    from temporalio.client import Client
    from temporalio.worker import Worker

    client = await Client.connect("localhost:7233")

    worker = Worker(
        client,
        task_queue="vectordb-queries",
        workflows=[QueryWorkflow],
        activities=[
            preprocess_query,
            generate_embedding,
            retrieve_chunks,
            rerank_results
        ]
    )

    await worker.run()
```

---

## Phase 10: Documentation & Demo (Priority: CRITICAL)

### Task 10.1: Comprehensive README
```markdown
# Vector Database REST API

A production-grade vector database with custom-implemented indexing algorithms.

## Features

- **Multiple Index Types**: Brute-force, K-D Tree, LSH
- **Thread-Safe**: Read-write locks for concurrent access
- **Persistent**: WAL + snapshots for durability
- **Metadata Filtering**: Query with complex filters
- **REST API**: Full CRUD operations
- **Python SDK**: Easy-to-use client library
- **Dockerized**: Ready for deployment

## Architecture

### Domain-Driven Design
```
domain/         # Business logic
├── models/     # Pydantic models
├── entities/   # Domain entities
└── exceptions/ # Custom exceptions

application/    # Use cases
└── services/   # Application services

infrastructure/ # Technical implementations
├── indexes/    # Vector indexes
├── repositories/ # Data access
└── persistence/ # Disk storage

api/            # FastAPI layer
└── routes/     # HTTP endpoints
```

### Indexing Algorithms

#### 1. Brute Force
- **Time**: O(n·d) search
- **Space**: O(n·d)
- **Use**: Small datasets (<10K vectors)
- **Accuracy**: Exact

#### 2. K-D Tree
- **Time**: O(log n) avg, O(n) worst
- **Space**: O(n·d)
- **Use**: Low dimensions (<20)
- **Accuracy**: Exact

#### 3. LSH (Locality-Sensitive Hashing)
- **Time**: O(n^ρ) for ρ < 1
- **Space**: O(n·L)
- **Use**: High dimensions, large datasets
- **Accuracy**: Approximate (tunable)

## Quick Start

### Docker
```bash
docker-compose up
```

### Local
```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
```

### Python SDK
```python
from vectordb_client import VectorDBClient

client = VectorDBClient("http://localhost:8000")
library = client.create_library("My Library")
client.add_document(library["id"], text="Document content")
results = client.query(library["id"], query_text="search", k=5)
```

## API Documentation

Interactive docs: http://localhost:8000/docs

### Key Endpoints

- `POST /libraries` - Create library
- `POST /libraries/{id}/documents` - Add document
- `POST /libraries/{id}/query` - Search
- `GET /libraries/{id}` - Get library

## Design Decisions

### Concurrency
- **Read-Write Locks**: Allow concurrent reads, exclusive writes
- **Writer Priority**: Prevent writer starvation
- **Async Support**: AsyncIO-compatible locks for FastAPI

### Persistence
- **WAL**: Append-only log for durability
- **Snapshots**: Periodic full state saves
- **Recovery**: Replay WAL from last snapshot

### Thread Safety
- All repository operations protected by locks
- Index operations atomic
- Exception safety via context managers

## Testing

```bash
pytest tests/ -v --cov=src
```

## Performance

Benchmarks on 100K vectors (768 dimensions):

| Index | Build Time | Query Time (k=10) | Recall@10 |
|-------|------------|-------------------|-----------|
| Brute | 1s         | 450ms             | 100%      |
| KD-Tree | 15s      | 380ms             | 100%      |
| LSH   | 5s         | 45ms              | 95%       |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

## License

MIT
```

### Task 10.2: Design Document
Create a separate `DESIGN.md` documenting:
- Architecture diagrams
- Algorithm explanations with complexity analysis
- Concurrency model
- Persistence strategy
- Trade-off discussions

### Task 10.3: Demo Videos
1. **Installation & Usage Demo** (5-10 min)
   - Clone repo
   - Docker setup
   - Create library via API
   - Add documents
   - Query and show results
   - Show Python SDK usage

2. **Design Walkthrough** (10-15 min)
   - Architecture overview
   - Index implementations
   - Thread-safety mechanisms
   - Persistence layer
   - Design trade-offs

---

## Implementation Timeline

### Week 1: Foundation
- Days 1-2: Domain models, exceptions, base classes
- Days 3-4: Index implementations (all three)
- Days 5-7: Testing indexes thoroughly

### Week 2: Services & API
- Days 1-2: Repository & concurrency
- Days 3-4: Service layer
- Days 5-6: FastAPI endpoints
- Day 7: Integration tests

### Week 3: Advanced Features
- Days 1-2: Persistence (WAL + snapshots)
- Days 3-4: Metadata filtering
- Days 5-6: Python SDK
- Day 7: Testing & bug fixes

### Week 4: Polish & Extras
- Days 1-2: Docker & deployment
- Days 3-4: Temporal workflows (optional)
- Days 5-6: Documentation
- Day 7: Demo videos

---

## Code Quality Checklist

### SOLID Principles
- [ ] Single Responsibility: Each class has one reason to change
- [ ] Open/Closed: Easy to add new index types without modifying existing
- [ ] Liskov Substitution: All indexes implement same interface
- [ ] Interface Segregation: Clients depend on minimal interfaces
- [ ] Dependency Inversion: Depend on abstractions, not concretions

### Best Practices
- [ ] Type hints everywhere
- [ ] Docstrings on all public methods
- [ ] Early returns to reduce nesting
- [ ] Context managers for resource management
- [ ] No hardcoded values (use constants/enums)
- [ ] Proper exception handling
- [ ] Logging at appropriate levels
- [ ] Tests covering edge cases
- [ ] FastAPI best practices (dependency injection, response models)
- [ ] Pydantic validation on all inputs

### Performance
- [ ] NumPy for vectorized operations
- [ ] Efficient data structures (dicts for O(1) lookup)
- [ ] Lazy loading where appropriate
- [ ] Caching for expensive operations
- [ ] Batch processing for embeddings

---

## Success Criteria

### Functionality
- [ ] All CRUD operations work
- [ ] All three indexes return correct results
- [ ] Query with filters works
- [ ] Persistence survives restart
- [ ] Thread-safety under load

### Code Quality
- [ ] 90%+ test coverage
- [ ] No mypy errors
- [ ] All linters pass
- [ ] Clean git history
- [ ] Comprehensive documentation

### Delivery
- [ ] GitHub repository with README
- [ ] Docker image builds successfully
- [ ] Demo videos recorded
- [ ] Design document complete

---

## Additional Enhancements (Time Permitting)

1. **Observability**
   - Structured logging (JSON logs)
   - Metrics (Prometheus)
   - Tracing (OpenTelemetry)

2. **Security**
   - API key authentication
   - Rate limiting
   - Input sanitization

3. **Performance**
   - Caching layer (Redis)
   - Query result caching
   - Index warming

4. **Scalability**
   - Horizontal scaling with Redis
   - Index sharding
   - Async background jobs

5. **UI Dashboard**
   - Web UI for library management
   - Query interface
   - Metrics visualization

---

This plan provides a comprehensive roadmap for building an impressive Vector Database REST API that demonstrates mastery of software engineering principles, algorithms, and system design.
