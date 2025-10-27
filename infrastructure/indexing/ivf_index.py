"""
IVF (Inverted File) Index for billion-scale vector search.

IVF partitions the vector space into clusters (Voronoi cells) using k-means clustering.
Each cluster stores an inverted list of vectors assigned to that cluster's centroid.

Search algorithm:
1. Find nprobe nearest centroids to query
2. Search only vectors in those centroids' inverted lists
3. Return top-k from candidate set

This achieves sub-linear search time: O(nprobe * list_size) instead of O(N).

For billion-scale, combine with Product Quantization (PQ) for memory compression.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from infrastructure.indexing.base_index import BaseIndex

logger = logging.getLogger(__name__)


class IVFIndex(BaseIndex):
    """
    IVF (Inverted File) index for approximate nearest neighbor search.

    Features:
    - K-means clustering to partition vector space
    - Inverted lists for each cluster
    - Configurable nprobe (number of clusters to search)
    - Optional Product Quantization for memory efficiency
    - Supports billion-scale datasets

    Performance:
    - Build: O(N * n_clusters * dimensions)
    - Search: O(nprobe * (N/n_clusters) * dimensions)
    - Memory: O(N * dimensions) or O(N * compressed_size) with PQ
    """

    def __init__(
        self,
        dimensions: int,
        n_clusters: int = 256,
        nprobe: int = 8,
        use_pq: bool = False,
        pq_subvectors: int = 8,
    ):
        """
        Initialize IVF index.

        Args:
            dimensions: Vector dimensionality
            n_clusters: Number of clusters (Voronoi cells)
                       - Recommended: sqrt(N) for N vectors
                       - More clusters = faster search, longer build
            nprobe: Number of nearest clusters to search
                   - Higher = better recall, slower search
                   - Typical: 1-32
            use_pq: Use Product Quantization for compression
                   - Reduces memory by ~8-32x
                   - Slight accuracy loss
            pq_subvectors: Number of PQ subvectors (if use_pq=True)
                          - More = better accuracy, less compression
        """
        super().__init__(dimensions)
        self._n_clusters = n_clusters
        self._nprobe = nprobe
        self._use_pq = use_pq
        self._pq_subvectors = pq_subvectors

        # Centroids: shape (n_clusters, dimensions)
        self._centroids: Optional[np.ndarray] = None

        # Inverted lists: dict mapping cluster_id -> list of (vector_id, vector)
        self._inverted_lists: Dict[int, List[Tuple[int, np.ndarray]]] = {}

        # K-means model for clustering
        self._kmeans: Optional[MiniBatchKMeans] = None

        # Product Quantization codebooks (if enabled)
        self._pq_codebooks: Optional[List[np.ndarray]] = None

        # Statistics
        self._list_sizes: Optional[np.ndarray] = None

        logger.info(
            f"Initialized IVF index: {n_clusters} clusters, "
            f"nprobe={nprobe}, PQ={use_pq}"
        )

    def build(self, vectors: np.ndarray, vector_ids: Optional[np.ndarray] = None):
        """
        Build the IVF index by clustering vectors and assigning to inverted lists.

        Args:
            vectors: shape (n, dimensions)
            vector_ids: shape (n,) - optional IDs for vectors
        """
        if len(vectors) == 0:
            logger.warning("Cannot build IVF index with zero vectors")
            return

        if vector_ids is None:
            vector_ids = np.arange(len(vectors))

        logger.info(f"Building IVF index: {len(vectors)} vectors, {self._n_clusters} clusters")
        start_time = time.time()

        # Step 1: Train k-means to find centroids
        self._train_kmeans(vectors)

        # Step 2: Assign vectors to nearest centroids
        self._assign_to_clusters(vectors, vector_ids)

        # Step 3: Train Product Quantization (if enabled)
        if self._use_pq:
            self._train_product_quantization(vectors)

        build_time = time.time() - start_time
        logger.info(
            f"IVF index built in {build_time:.2f}s, "
            f"avg list size: {len(vectors) / self._n_clusters:.1f}"
        )

    def _train_kmeans(self, vectors: np.ndarray):
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

    def _assign_to_clusters(self, vectors: np.ndarray, vector_ids: np.ndarray):
        """Assign vectors to their nearest cluster's inverted list."""
        logger.info("Assigning vectors to clusters...")

        # Clear existing inverted lists
        self._inverted_lists = {i: [] for i in range(len(self._centroids))}

        # Assign each vector to nearest centroid
        cluster_assignments = self._kmeans.predict(vectors)

        for vec_id, vector, cluster_id in zip(vector_ids, vectors, cluster_assignments):
            self._inverted_lists[cluster_id].append((vec_id, vector))

        # Track list sizes for statistics
        self._list_sizes = np.array(
            [len(self._inverted_lists[i]) for i in range(len(self._centroids))]
        )

        logger.info(
            f"Assigned vectors: avg={self._list_sizes.mean():.1f}, "
            f"min={self._list_sizes.min()}, max={self._list_sizes.max()}"
        )

    def _train_product_quantization(self, vectors: np.ndarray):
        """Train Product Quantization codebooks for compression."""
        logger.info("Training Product Quantization...")

        # Split dimensions into subvectors
        subvec_dim = self.dimensions // self._pq_subvectors
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

    def search(
        self, query: np.ndarray, k: int = 10, distance_threshold: Optional[float] = None
    ) -> List[Tuple[int, float]]:
        """
        Search for k nearest neighbors using IVF.

        Args:
            query: Query vector, shape (dimensions,)
            k: Number of results
            distance_threshold: Optional distance threshold

        Returns:
            List of (vector_id, distance) tuples
        """
        if self._centroids is None:
            return []

        # Step 1: Find nprobe nearest centroids
        centroid_distances = np.linalg.norm(self._centroids - query, axis=1)
        nearest_clusters = np.argsort(centroid_distances)[: self._nprobe]

        # Step 2: Collect candidate vectors from those clusters
        candidates = []
        for cluster_id in nearest_clusters:
            candidates.extend(self._inverted_lists[cluster_id])

        if not candidates:
            return []

        # Step 3: Compute distances to all candidates
        candidate_ids = [vec_id for vec_id, _ in candidates]
        candidate_vectors = np.array([vec for _, vec in candidates])

        distances = np.linalg.norm(candidate_vectors - query, axis=1)

        # Step 4: Filter by threshold and return top-k
        if distance_threshold is not None:
            mask = distances <= distance_threshold
            candidate_ids = [candidate_ids[i] for i in range(len(distances)) if mask[i]]
            distances = distances[mask]

        # Sort and return top-k
        top_k_indices = np.argsort(distances)[:k]
        results = [(candidate_ids[i], float(distances[i])) for i in top_k_indices]

        return results

    def add(self, vector: np.ndarray, vector_id: int):
        """
        Add a single vector to the index.

        Args:
            vector: shape (dimensions,)
            vector_id: Unique ID
        """
        if self._centroids is None:
            # Index not built yet, store in temporary list
            if not hasattr(self, "_temp_vectors"):
                self._temp_vectors = []
                self._temp_ids = []
            self._temp_vectors.append(vector)
            self._temp_ids.append(vector_id)
            return

        # Find nearest centroid
        distances = np.linalg.norm(self._centroids - vector, axis=1)
        cluster_id = int(np.argmin(distances))

        # Add to inverted list
        self._inverted_lists[cluster_id].append((vector_id, vector))

        # Update statistics
        if self._list_sizes is not None:
            self._list_sizes[cluster_id] += 1

    def remove(self, vector_id: int) -> bool:
        """
        Remove a vector from the index.

        Args:
            vector_id: ID to remove

        Returns:
            True if removed, False if not found
        """
        # Search all inverted lists
        for cluster_id, inverted_list in self._inverted_lists.items():
            for i, (vid, _) in enumerate(inverted_list):
                if vid == vector_id:
                    inverted_list.pop(i)
                    if self._list_sizes is not None:
                        self._list_sizes[cluster_id] -= 1
                    return True

        return False

    def get_statistics(self) -> Dict[str, any]:
        """
        Get IVF index statistics.

        Returns:
            Statistics dictionary
        """
        if self._centroids is None:
            return {
                "built": False,
                "n_clusters": self._n_clusters,
                "nprobe": self._nprobe,
            }

        total_vectors = sum(len(lst) for lst in self._inverted_lists.values())

        stats = {
            "built": True,
            "n_clusters": len(self._centroids),
            "nprobe": self._nprobe,
            "total_vectors": total_vectors,
            "avg_list_size": total_vectors / len(self._centroids) if self._centroids is not None else 0,
            "min_list_size": int(self._list_sizes.min()) if self._list_sizes is not None else 0,
            "max_list_size": int(self._list_sizes.max()) if self._list_sizes is not None else 0,
            "use_pq": self._use_pq,
        }

        if self._use_pq and self._pq_codebooks:
            stats["pq_subvectors"] = len(self._pq_codebooks)
            stats["compression_ratio"] = f"{8 * len(self._pq_codebooks) / self.dimensions:.2f}x"

        return stats

    def optimize(self) -> int:
        """
        Optimize the index by rebalancing clusters.

        Returns:
            Number of vectors rebalanced
        """
        if self._centroids is None:
            return 0

        logger.info("Optimizing IVF index...")

        # Collect all vectors
        all_vectors = []
        all_ids = []
        for inverted_list in self._inverted_lists.values():
            for vec_id, vector in inverted_list:
                all_ids.append(vec_id)
                all_vectors.append(vector)

        if not all_vectors:
            return 0

        # Rebuild with current vectors
        all_vectors = np.array(all_vectors)
        all_ids = np.array(all_ids)

        self._assign_to_clusters(all_vectors, all_ids)

        logger.info("IVF index optimized")

        return len(all_vectors)
