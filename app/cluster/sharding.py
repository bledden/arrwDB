"""
Consistent Hash Sharding for arrwDB.

Distributes vectors across N shards using consistent hashing.
Each shard is an independent arrwDB instance (local or remote).

Architecture:
  Router → hash(vector_id) → shard_N → local index or remote HTTP call

Configuration:
  SHARDING_ENABLED=true
  SHARD_COUNT=4
  SHARD_URLS=http://shard0:8000,http://shard1:8000,...  (for remote shards)

For local sharding (single machine, multiple indexes):
  SHARDING_MODE=local
  SHARD_COUNT=4
  Each shard gets its own FastHNSW index in the same process.

For distributed sharding (multiple machines):
  SHARDING_MODE=distributed
  SHARD_URLS=...
  Requests are routed to the correct shard via HTTP.
"""

import hashlib
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class ConsistentHashRing:
    """Consistent hash ring for shard assignment.

    Uses virtual nodes (replicas) for even distribution.
    Adding/removing a shard only remaps ~1/N of the keys.
    """

    def __init__(self, num_shards: int, virtual_nodes: int = 150):
        self.num_shards = num_shards
        self.virtual_nodes = virtual_nodes
        self._ring: list[Tuple[int, int]] = []  # (hash_value, shard_id)
        self._build_ring()

    def _build_ring(self):
        self._ring.clear()
        for shard_id in range(self.num_shards):
            for vn in range(self.virtual_nodes):
                key = f"shard-{shard_id}-vn-{vn}"
                h = self._hash(key)
                self._ring.append((h, shard_id))
        self._ring.sort(key=lambda x: x[0])

    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def get_shard(self, key: str) -> int:
        """Get the shard ID for a given key (vector ID, document ID, etc.)."""
        h = self._hash(key)

        # Binary search for the first ring position >= h
        lo, hi = 0, len(self._ring) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._ring[mid][0] < h:
                lo = mid + 1
            else:
                hi = mid

        return self._ring[lo][1]

    def get_shard_for_query(self) -> List[int]:
        """For search queries, return ALL shard IDs (scatter-gather)."""
        return list(range(self.num_shards))


class ShardRouter:
    """Routes requests to the correct shard.

    For writes: hash the document/vector ID → single shard.
    For searches: scatter to all shards → gather and merge results.
    """

    def __init__(self, num_shards: int, shard_urls: Optional[List[str]] = None):
        self.ring = ConsistentHashRing(num_shards)
        self.num_shards = num_shards
        self.shard_urls = shard_urls or []
        self.local_mode = len(self.shard_urls) == 0

        logger.info(
            f"ShardRouter: {num_shards} shards, "
            f"mode={'local' if self.local_mode else 'distributed'}"
        )

    def route_write(self, key: str) -> int:
        """Route a write operation to the correct shard."""
        return self.ring.get_shard(key)

    def route_search(self) -> List[int]:
        """Route a search to all shards (scatter-gather)."""
        return self.ring.get_shard_for_query()

    def get_shard_url(self, shard_id: int) -> Optional[str]:
        """Get the URL for a remote shard."""
        if shard_id < len(self.shard_urls):
            return self.shard_urls[shard_id]
        return None

    def merge_search_results(
        self,
        shard_results: List[List[Tuple[str, float]]],
        k: int,
    ) -> List[Tuple[str, float]]:
        """Merge search results from multiple shards, keeping top-k by distance."""
        all_results = []
        for results in shard_results:
            all_results.extend(results)

        # Sort by distance (ascending = most similar first)
        all_results.sort(key=lambda x: x[1])
        return all_results[:k]
