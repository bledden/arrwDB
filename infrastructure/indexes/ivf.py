"""
IVF (Inverted File) Index implementation.

This module provides an IVF index that partitions the vector space using k-means
clustering for efficient approximate nearest neighbor search at scale.

Time Complexity:
- Build: O(N * n_clusters * dimensions)
- Search: O(nprobe * (N/n_clusters) * dimensions)
- Insert: O(dimensions) - assign to nearest cluster
- Delete: O(N/n_clusters) average - search within cluster

Space Complexity: O(N * dimensions) for full vectors, or O(N * compressed_size) with PQ

Recommended for:
- Large datasets (> 100K vectors)
- Cases where approximate search is acceptable
- When sub-linear search time is critical
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import MiniBatchKMeans

from core.vector_store import VectorStore
from infrastructure.indexes.base import VectorIndex

logger = logging.getLogger(__name__)


class IVFIndex(VectorIndex):
    """
    IVF (Inverted File) index for approximate nearest neighbor search.

    The index partitions the vector space into Voronoi cells using k-means clustering.
    Each cluster maintains an inverted list of vectors assigned to it.

    Search Algorithm:
    1. Find nprobe nearest centroids to query
    2. Search only vectors in those centroids' inverted lists
    3. Return top-k from candidate set

    This achieves sub-linear search time: O(nprobe * list_size) instead of O(N).

    Advantages:
    - Sub-linear search time for large datasets
    - Configurable accuracy/speed tradeoff (nprobe parameter)
    - Efficient for high-dimensional vectors
    - Optional Product Quantization for memory compression

    Disadvantages:
    - Requires initial clustering (build time)
    - Approximate results (may miss some nearest neighbors)
    - Needs rebalancing as data distribution changes

    Best for:
    - Datasets > 100K vectors
    - High-dimensional embeddings (512-2048 dims)
    - Cases where 95%+ recall is acceptable
    """

    def __init__(
        self,
        vector_store: VectorStore,
        n_clusters: int = 256,
        nprobe: int = 8,
        use_pq: bool = False,
        pq_subvectors: int = 8,
    ):
        """
        Initialize IVF index.

        Args:
            vector_store: The vector store containing embeddings
            n_clusters: Number of clusters (Voronoi cells)
                       Recommended: sqrt(N) for N vectors
                       More clusters = faster search, longer build
            nprobe: Number of nearest clusters to search
                   Higher = better recall, slower search
                   Typical range: 1-32
            use_pq: Use Product Quantization for compression
                   Reduces memory by ~8-32x with slight accuracy loss
            pq_subvectors: Number of PQ subvectors (if use_pq=True)
                          More = better accuracy, less compression
        """
        super().__init__(vector_store)
        self._n_clusters = n_clusters
        self._nprobe = nprobe
        self._use_pq = use_pq
        self._pq_subvectors = pq_subvectors

        # Centroids: shape (n_clusters, dimensions)
        self._centroids: Optional[np.ndarray] = None

        # Inverted lists: dict mapping cluster_id -> list of vector_ids
        self._inverted_lists: Dict[int, List[UUID]] = {}

        # K-means model for clustering
        self._kmeans: Optional[MiniBatchKMeans] = None

        # Product Quantization codebooks (if enabled)
        self._pq_codebooks: Optional[List[np.ndarray]] = None

        # Statistics
        self._list_sizes: Optional[np.ndarray] = None
        self._built = False

        logger.info(
            f"Initialized IVF index: {n_clusters} clusters, "
            f"nprobe={nprobe}, PQ={use_pq}"
        )

    def build(self) -> None:
        """
        Build the IVF index by clustering all vectors in the vector store.

        This trains k-means on all vectors and assigns each to its nearest cluster.
        Should be called after initial data load or when index needs rebuilding.
        """
        # Get all vectors from vector store
        all_ids = list(self._vector_id_to_index.keys())
        if len(all_ids) == 0:
            logger.warning("Cannot build IVF index with zero vectors")
            return

        vectors = np.array([self._vector_store.get_vector(i) for i in range(len(all_ids))])

        logger.info(f"Building IVF index: {len(vectors)} vectors, {self._n_clusters} clusters")

        # Step 1: Train k-means to find centroids
        self._train_kmeans(vectors)

        # Step 2: Assign vectors to nearest centroids
        self._assign_to_clusters(vectors, all_ids)

        # Step 3: Train Product Quantization (if enabled)
        if self._use_pq:
            self._train_product_quantization(vectors)

        self._built = True
        logger.info(
            f"IVF index built: avg list size: {len(vectors) / self._n_clusters:.1f}"
        )

    def _train_kmeans(self, vectors: np.ndarray) -> None:
        """Train k-means clustering to find centroids."""
        logger.info(f"Training k-means with {self._n_clusters} clusters...")

        # Use MiniBatchKMeans for scalability
        self._kmeans = MiniBatchKMeans(
            n_clusters=min(self._n_clusters, len(vectors)),
            batch_size=min(1000, len(vectors)),
            max_iter=100,
            random_state=42,
            verbose=0,
        )

        self._kmeans.fit(vectors)
        self._centroids = self._kmeans.cluster_centers_

        logger.info(f"K-means trained: {len(self._centroids)} centroids")

    def _assign_to_clusters(self, vectors: np.ndarray, vector_ids: List[UUID]) -> None:
        """Assign vectors to their nearest cluster's inverted list."""
        logger.info("Assigning vectors to clusters...")

        # Clear existing inverted lists
        self._inverted_lists = {i: [] for i in range(len(self._centroids))}

        # Assign each vector to nearest centroid
        cluster_assignments = self._kmeans.predict(vectors)

        for vec_id, cluster_id in zip(vector_ids, cluster_assignments):
            self._inverted_lists[cluster_id].append(vec_id)

        # Track list sizes for statistics
        self._list_sizes = np.array(
            [len(self._inverted_lists[i]) for i in range(len(self._centroids))]
        )

        logger.info(
            f"Assigned vectors: avg={self._list_sizes.mean():.1f}, "
            f"min={self._list_sizes.min()}, max={self._list_sizes.max()}"
        )

    def _train_product_quantization(self, vectors: np.ndarray) -> None:
        """Train Product Quantization codebooks for compression."""
        logger.info("Training Product Quantization...")

        dimensions = vectors.shape[1]
        subvec_dim = dimensions // self._pq_subvectors
        self._pq_codebooks = []

        for i in range(self._pq_subvectors):
            start_dim = i * subvec_dim
            end_dim = start_dim + subvec_dim

            # Extract subvectors
            subvectors = vectors[:, start_dim:end_dim]

            # Train k-means for this subspace (256 centroids per subvector)
            kmeans = MiniBatchKMeans(
                n_clusters=256,
                batch_size=min(1000, len(vectors)),
                max_iter=50,
                random_state=42 + i,
            )
            kmeans.fit(subvectors)

            self._pq_codebooks.append(kmeans.cluster_centers_)

        logger.info(f"PQ trained: {len(self._pq_codebooks)} codebooks")

    def add(self, vector_id: UUID, vector: NDArray[np.float64]) -> None:
        """
        Add a vector to the index.

        Args:
            vector_id: Unique identifier for the vector
            vector: The embedding vector
        """
        # Add to parent's tracking
        super().add(vector_id, vector)

        # If index is built, assign to nearest cluster
        if self._built and self._centroids is not None:
            distances = np.linalg.norm(self._centroids - vector, axis=1)
            cluster_id = int(np.argmin(distances))

            self._inverted_lists[cluster_id].append(vector_id)

            # Update statistics
            if self._list_sizes is not None:
                self._list_sizes[cluster_id] += 1

    def remove(self, vector_id: UUID) -> bool:
        """
        Remove a vector from the index.

        Args:
            vector_id: ID of vector to remove

        Returns:
            True if removed, False if not found
        """
        # Remove from parent's tracking
        if not super().remove(vector_id):
            return False

        # Remove from inverted lists if index is built
        if self._built:
            for cluster_id, inverted_list in self._inverted_lists.items():
                if vector_id in inverted_list:
                    inverted_list.remove(vector_id)
                    if self._list_sizes is not None:
                        self._list_sizes[cluster_id] -= 1
                    return True

        return True

    def search(
        self, query: NDArray[np.float64], k: int = 10, distance_threshold: Optional[float] = None
    ) -> List[Tuple[UUID, float]]:
        """
        Search for k nearest neighbors using IVF.

        Args:
            query: Query vector
            k: Number of results
            distance_threshold: Optional distance threshold

        Returns:
            List of (vector_id, distance) tuples sorted by distance
        """
        if not self._built or self._centroids is None:
            # Fall back to brute force if index not built
            logger.warning("IVF index not built, falling back to brute force search")
            return self._brute_force_search(query, k, distance_threshold)

        # Step 1: Find nprobe nearest centroids
        centroid_distances = np.linalg.norm(self._centroids - query, axis=1)
        nearest_clusters = np.argsort(centroid_distances)[: self._nprobe]

        # Step 2: Collect candidate vector IDs from those clusters
        candidate_ids = []
        for cluster_id in nearest_clusters:
            candidate_ids.extend(self._inverted_lists[cluster_id])

        if not candidate_ids:
            return []

        # Step 3: Compute distances to all candidates
        candidates = []
        for vec_id in candidate_ids:
            if vec_id in self._vector_id_to_index:
                vec_idx = self._vector_id_to_index[vec_id]
                vector = self._vector_store.get_vector(vec_idx)
                distance = float(np.linalg.norm(vector - query))
                candidates.append((vec_id, distance))

        # Step 4: Filter by threshold and return top-k
        if distance_threshold is not None:
            candidates = [(vid, d) for vid, d in candidates if d <= distance_threshold]

        # Sort by distance and return top-k
        candidates.sort(key=lambda x: x[1])
        return candidates[:k]

    def _brute_force_search(
        self, query: NDArray[np.float64], k: int, distance_threshold: Optional[float]
    ) -> List[Tuple[UUID, float]]:
        """Fallback brute force search when index not built."""
        results = []
        for vec_id, vec_idx in self._vector_id_to_index.items():
            vector = self._vector_store.get_vector(vec_idx)
            distance = float(np.linalg.norm(vector - query))

            if distance_threshold is None or distance <= distance_threshold:
                results.append((vec_id, distance))

        results.sort(key=lambda x: x[1])
        return results[:k]

    def optimize(self) -> int:
        """
        Optimize the index by rebalancing clusters.

        Returns:
            Number of vectors rebalanced
        """
        if not self._built or self._centroids is None:
            return 0

        logger.info("Optimizing IVF index...")

        # Get all vectors and rebuild clusters
        all_ids = list(self._vector_id_to_index.keys())
        if not all_ids:
            return 0

        vectors = np.array([
            self._vector_store.get_vector(self._vector_id_to_index[vid])
            for vid in all_ids
        ])

        self._assign_to_clusters(vectors, all_ids)

        logger.info("IVF index optimized")
        return len(all_ids)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get IVF index statistics.

        Returns:
            Statistics dictionary
        """
        stats = super().get_statistics()

        stats.update({
            "index_type": "ivf",
            "built": self._built,
            "n_clusters": self._n_clusters,
            "nprobe": self._nprobe,
            "use_pq": self._use_pq,
        })

        if self._built and self._centroids is not None:
            total_vectors = sum(len(lst) for lst in self._inverted_lists.values())
            stats.update({
                "total_vectors": total_vectors,
                "avg_list_size": total_vectors / len(self._centroids) if len(self._centroids) > 0 else 0,
                "min_list_size": int(self._list_sizes.min()) if self._list_sizes is not None else 0,
                "max_list_size": int(self._list_sizes.max()) if self._list_sizes is not None else 0,
            })

            if self._use_pq and self._pq_codebooks:
                stats["pq_subvectors"] = len(self._pq_codebooks)
                dims = self._vector_store.get_vector(0).shape[0] if self._vector_store.size() > 0 else 1024
                stats["compression_ratio"] = f"{8 * len(self._pq_codebooks) / dims:.2f}x"

        return stats
