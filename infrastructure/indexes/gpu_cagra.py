"""
GPU CAGRA Index via FAISS-GPU (cuVS backend).

CAGRA (CUDA Approximate Nearest Neighbors Graph-based) is NVIDIA's
graph-based ANN algorithm designed from scratch for GPU parallelism.
Unlike HNSW (sequential layer traversal), CAGRA uses a flat, fixed-degree
graph with coalesced memory access across GPU warps.

Performance vs CPU HNSW:
- Build: 10-12x faster
- Search: 33-77x faster at equivalent recall
- Can convert CAGRA graph to HNSW for CPU-only serving

Requires: conda install -c pytorch -c nvidia -c rapidsai -c conda-forge faiss-gpu-cuvs
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from core.vector_store import VectorStore
from infrastructure.indexes.base import VectorIndex

logger = logging.getLogger(__name__)


class GPUCagraIndex(VectorIndex):
    """GPU-accelerated CAGRA index via FAISS-GPU.

    CAGRA builds a flat, fixed-degree graph optimized for GPU parallelism.
    All vectors must be added before searching (batch-oriented, not incremental).
    Call rebuild() after adding vectors to construct the GPU index.

    Args:
        vector_store: VectorStore containing the vectors.
        graph_degree: Number of neighbors per node in the final graph (default: 64).
        intermediate_graph_degree: Graph degree during construction (default: 128).
        itopk_size: Search accuracy knob — higher = better recall, slower (default: 128).
        metric: Distance metric — "l2" or "inner_product" (default: "inner_product").
    """

    def __init__(
        self,
        vector_store: VectorStore,
        graph_degree: int = 64,
        intermediate_graph_degree: int = 128,
        itopk_size: int = 128,
        metric: str = "inner_product",
    ):
        try:
            import faiss

            if not hasattr(faiss, "GpuIndexCagra"):
                raise ImportError("faiss-gpu-cuvs required (has GpuIndexCagra)")
            self._faiss = faiss
        except ImportError:
            raise ImportError(
                "GPU CAGRA requires faiss-gpu-cuvs. Install with:\n"
                "  conda install -c pytorch -c nvidia -c rapidsai -c conda-forge faiss-gpu-cuvs"
            )

        self._vector_store = vector_store
        self._dimension = vector_store.dimension
        self._graph_degree = graph_degree
        self._intermediate_graph_degree = intermediate_graph_degree
        self._itopk_size = itopk_size
        self._metric = metric

        self._id_to_idx: Dict[UUID, int] = {}
        self._idx_to_id: Dict[int, UUID] = {}
        self._next_idx: int = 0
        self._gpu_index = None
        self._gpu_res = None
        self._is_built = False

        logger.info(
            f"GPUCagraIndex: dim={self._dimension}, "
            f"graph_degree={graph_degree}, metric={metric}"
        )

    def add_vector(self, vector_id: UUID, vector_index: int) -> None:
        if vector_id in self._id_to_idx:
            raise ValueError(f"Vector {vector_id} already exists")

        idx = self._next_idx
        self._id_to_idx[vector_id] = idx
        self._idx_to_id[idx] = vector_id
        self._next_idx += 1
        self._is_built = False  # Index needs rebuild

    def remove_vector(self, vector_id: UUID) -> bool:
        if vector_id not in self._id_to_idx:
            return False
        idx = self._id_to_idx.pop(vector_id)
        self._idx_to_id.pop(idx, None)
        self._is_built = False
        return True

    def search(
        self,
        query_vector: NDArray[np.float32],
        k: int,
        distance_threshold: Optional[float] = None,
    ) -> List[Tuple[UUID, float]]:
        if not self._is_built or self._gpu_index is None:
            if self._next_idx > 0:
                self.rebuild()
            else:
                return []

        query = query_vector.reshape(1, -1).astype(np.float32)
        D, I = self._gpu_index.search(query, k)

        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            if distance_threshold is not None and dist > distance_threshold:
                continue
            uid = self._idx_to_id.get(int(idx))
            if uid is not None:
                results.append((uid, float(dist)))

        return results

    def rebuild(self) -> None:
        if self._next_idx == 0:
            self._gpu_index = None
            self._is_built = True
            return

        # Collect all vectors from the store
        active_ids = sorted(self._id_to_idx.items(), key=lambda x: x[1])
        indices = [self._id_to_idx[uid] for uid, _ in active_ids]

        # Get vectors in order
        vectors = []
        for uid, idx in active_ids:
            vec = self._vector_store.get_vector(uid)
            if vec is not None:
                vectors.append(vec)

        if not vectors:
            self._gpu_index = None
            self._is_built = True
            return

        data = np.vstack(vectors).astype(np.float32)
        n, d = data.shape

        logger.info(f"Building GPU CAGRA index: {n} vectors, dim={d}")
        start = time.time()

        # Select metric
        if self._metric == "inner_product":
            metric = self._faiss.METRIC_INNER_PRODUCT
        else:
            metric = self._faiss.METRIC_L2

        # Configure CAGRA
        config = self._faiss.GpuIndexCagraConfig()
        config.graph_degree = self._graph_degree
        config.intermediate_graph_degree = self._intermediate_graph_degree

        # GPU resources
        self._gpu_res = self._faiss.StandardGpuResources()
        self._gpu_index = self._faiss.GpuIndexCagra(
            self._gpu_res, d, metric, config
        )
        self._gpu_index.train(data)
        self._gpu_index.add(data)

        build_time = time.time() - start
        logger.info(
            f"GPU CAGRA build complete: {build_time:.2f}s "
            f"({n / build_time:.0f} vec/s)"
        )
        self._is_built = True

    def size(self) -> int:
        return len(self._id_to_idx)

    def clear(self) -> None:
        self._id_to_idx.clear()
        self._idx_to_id.clear()
        self._next_idx = 0
        self._gpu_index = None
        self._gpu_res = None
        self._is_built = False

    @property
    def supports_incremental_updates(self) -> bool:
        return False  # CAGRA requires full rebuild

    @property
    def index_type(self) -> str:
        return "gpu_cagra"

    def set_itopk_size(self, itopk_size: int) -> None:
        """Set search accuracy parameter (higher = better recall, slower)."""
        self._itopk_size = itopk_size
        # FAISS-GPU CAGRA uses search params at search time via index params
        if self._gpu_index is not None:
            sp = self._faiss.SearchParametersCagra()
            sp.itopk_size = itopk_size
            self._search_params = sp

    def to_cpu_hnsw(self, M: int = 32) -> "faiss.IndexHNSW":
        """Convert GPU CAGRA graph to CPU HNSW index for serving without GPU."""
        if not self._is_built or self._gpu_index is None:
            raise RuntimeError("Index must be built before conversion")

        if self._metric == "inner_product":
            metric = self._faiss.METRIC_INNER_PRODUCT
        else:
            metric = self._faiss.METRIC_L2

        cpu_index = self._faiss.IndexHNSWCagra(self._dimension, M, metric)
        self._gpu_index.copyTo(cpu_index)
        logger.info(f"Converted GPU CAGRA to CPU HNSW (M={M})")
        return cpu_index
