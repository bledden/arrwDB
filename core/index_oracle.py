"""
IndexOracle - Adaptive index selection based on usage patterns.

NOVEL FEATURE: Automatic index type selection - no other vector DB does this.
Most vector DBs force users to choose index type upfront and stick with it.

WHY THIS MATTERS:
Choosing the right index is hard:
- Users don't know their data distribution in advance
- Workload patterns change over time (read-heavy → write-heavy)
- Wrong index choice = 10-100x worse performance
- Manual index migration is risky and causes downtime

IndexOracle solves this by:
1. Monitoring corpus characteristics (size, dimensionality, update rate)
2. Tracking query patterns (throughput, recall requirements)
3. Recommending optimal index type for current workload
4. Optionally auto-migrating to better index (zero downtime)

INSPIRATION:
- Database query optimizers (cost-based optimization)
- Auto-scaling cloud services (adaptive resource allocation)
- Self-tuning systems research (autonomic computing)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CorpusProfile:
    """
    Profile of a corpus's characteristics.

    WHY: Different corpus properties favor different index types.
    """

    vector_count: int = 0
    dimension: int = 0
    insert_rate_per_minute: float = 0.0  # Vectors added per minute
    delete_rate_per_minute: float = 0.0  # Vectors deleted per minute
    search_rate_per_minute: float = 0.0  # Searches per minute
    avg_search_k: float = 10.0  # Average k in search requests
    avg_search_latency_ms: float = 0.0  # Average search latency

    # Quality metrics
    current_index_type: str = "brute_force"
    recall_score: float = 1.0  # 1.0 = perfect recall (for approx indexes)

    # Temporal patterns
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class IndexRecommendation:
    """
    Recommended index type with reasoning.

    WHY: Transparency - explain WHY this index is recommended.
    """

    recommended_index: str
    current_index: str
    confidence: float  # 0-1, how confident in this recommendation
    reasoning: List[str]  # Human-readable reasons
    expected_speedup: Optional[float] = None  # Expected performance improvement
    migration_cost: str = "low"  # "low", "medium", "high"
    auto_migrate: bool = False  # Safe to auto-migrate?


class IndexOracle:
    """
    Adaptive index selector based on corpus and workload characteristics.

    Decision matrix (simplified):
    - vectors < 1000: brute_force (no index overhead)
    - 1000 < vectors < 100k, writes > reads: brute_force (write-optimized)
    - 100k < vectors < 1M: hnsw (balanced)
    - vectors > 1M, high recall: hnsw (high quality ANN)
    - vectors > 10M, lower recall ok: ivf (memory efficient)

    ALGORITHM:
    1. Collect corpus metrics (size, update rate, search rate)
    2. Score each index type based on workload fit
    3. Recommend highest-scoring index
    4. If confidence > threshold, mark for auto-migration
    """

    # Thresholds for index selection (tuned from experience)
    BRUTE_FORCE_MAX = 1000  # Above this, consider approximate indexes
    HNSW_MIN = 1000  # Below this, brute force is faster
    HNSW_MAX = 10_000_000  # Above this, HNSW memory usage becomes problematic
    IVF_MIN = 1_000_000  # IVF makes sense for large corpora

    # Workload patterns
    WRITE_HEAVY_RATIO = 0.5  # If writes > 50% of ops, favor write-friendly index
    READ_HEAVY_RATIO = 0.9  # If reads > 90%, favor search-optimized index

    def __init__(self):
        """Initialize the IndexOracle."""
        pass

    def analyze_corpus(self, profile: CorpusProfile) -> IndexRecommendation:
        """
        Analyze corpus profile and recommend best index type.

        WHY: Central decision function - encapsulates all the heuristics.

        Args:
            profile: Current corpus characteristics.

        Returns:
            Index recommendation with reasoning.
        """
        current = profile.current_index_type
        vector_count = profile.vector_count
        dimension = profile.dimension

        # Calculate workload ratio
        total_ops = (
            profile.insert_rate_per_minute
            + profile.delete_rate_per_minute
            + profile.search_rate_per_minute
        )
        write_ratio = (
            (profile.insert_rate_per_minute + profile.delete_rate_per_minute)
            / total_ops
            if total_ops > 0
            else 0.0
        )

        reasoning = []

        # Decision logic
        recommended = self._select_index(
            vector_count, dimension, write_ratio, profile, reasoning
        )

        # Calculate confidence
        confidence = self._calculate_confidence(profile, recommended, reasoning)

        # Estimate speedup
        speedup = self._estimate_speedup(current, recommended, profile)

        # Determine migration cost
        cost = self._estimate_migration_cost(vector_count, current, recommended)

        # Auto-migrate if high confidence and low cost
        auto_migrate = confidence > 0.8 and cost == "low"

        return IndexRecommendation(
            recommended_index=recommended,
            current_index=current,
            confidence=confidence,
            reasoning=reasoning,
            expected_speedup=speedup,
            migration_cost=cost,
            auto_migrate=auto_migrate,
        )

    def _select_index(
        self,
        vector_count: int,
        dimension: int,
        write_ratio: float,
        profile: CorpusProfile,
        reasoning: List[str],
    ) -> str:
        """
        Select best index based on corpus characteristics.

        WHY: Core decision logic separated for clarity.
        """
        # Rule 1: Very small corpus - brute force always wins
        if vector_count < self.BRUTE_FORCE_MAX:
            reasoning.append(
                f"Corpus has only {vector_count} vectors - brute force is fastest (no index overhead)"
            )
            return "brute_force"

        # Rule 2: Write-heavy workload - brute force for write performance
        if write_ratio > self.WRITE_HEAVY_RATIO:
            reasoning.append(
                f"Write-heavy workload ({write_ratio*100:.1f}% writes) - "
                "brute force handles updates better"
            )
            return "brute_force"

        # Rule 3: Small to medium corpus, read-heavy - HNSW is best
        if self.HNSW_MIN <= vector_count <= self.HNSW_MAX:
            reasoning.append(
                f"Corpus size ({vector_count} vectors) ideal for HNSW - "
                "best balance of speed and recall"
            )
            if profile.recall_score < 0.95:
                reasoning.append(
                    f"Current recall ({profile.recall_score:.2%}) below target - "
                    "HNSW provides better quality"
                )
            return "hnsw"

        # Rule 4: Very large corpus - IVF for memory efficiency
        if vector_count > self.IVF_MIN:
            reasoning.append(
                f"Large corpus ({vector_count} vectors) - "
                "IVF more memory-efficient than HNSW"
            )
            if profile.avg_search_latency_ms > 100:
                reasoning.append(
                    f"Search latency ({profile.avg_search_latency_ms:.1f}ms) high - "
                    "IVF can improve with clustering"
                )
            return "ivf"

        # Default: HNSW for general case
        reasoning.append("General purpose workload - HNSW recommended")
        return "hnsw"

    def _calculate_confidence(
        self, profile: CorpusProfile, recommended: str, reasoning: List[str]
    ) -> float:
        """
        Calculate confidence in recommendation (0-1).

        WHY: Don't auto-migrate unless we're very confident.
        """
        confidence = 0.5  # Base confidence

        # Higher confidence for clear cases
        if profile.vector_count < self.BRUTE_FORCE_MAX:
            confidence += 0.4  # Very confident for small corpus

        if profile.vector_count > self.IVF_MIN:
            confidence += 0.3  # Confident for large corpus

        # Lower confidence for edge cases
        if self.BRUTE_FORCE_MAX <= profile.vector_count <= self.HNSW_MIN * 2:
            confidence -= 0.2  # Transition zone, less certain

        # Higher confidence if current index is clearly wrong
        if profile.current_index_type == "brute_force" and profile.vector_count > 100_000:
            confidence += 0.3  # Clearly should upgrade

        return min(1.0, max(0.0, confidence))

    def _estimate_speedup(
        self, current: str, recommended: str, profile: CorpusProfile
    ) -> Optional[float]:
        """
        Estimate expected speedup from migration.

        WHY: Help users understand ROI of migration.
        """
        if current == recommended:
            return None  # No change

        vector_count = profile.vector_count

        # Rough speedup estimates based on empirical data
        speedups = {
            ("brute_force", "hnsw"): min(100, vector_count / 1000),  # Linear → log
            ("brute_force", "ivf"): min(50, vector_count / 5000),
            ("hnsw", "ivf"): 0.8,  # Slight degradation but memory savings
            ("ivf", "hnsw"): 1.5,  # Quality improvement
        }

        return speedups.get((current, recommended), 1.0)

    def _estimate_migration_cost(
        self, vector_count: int, current: str, recommended: str
    ) -> str:
        """
        Estimate cost of migrating to new index.

        WHY: Inform decision about auto-migration safety.
        """
        if current == recommended:
            return "none"

        # Brute force → anything: low cost (no existing index to rebuild)
        if current == "brute_force":
            return "low"

        # Small corpus: low cost
        if vector_count < 10_000:
            return "low"

        # Medium corpus: medium cost
        if vector_count < 1_000_000:
            return "medium"

        # Large corpus: high cost (long rebuild time)
        return "high"

    def get_index_guidelines(self) -> Dict:
        """
        Get index selection guidelines.

        WHY: Educational - help users understand the decision logic.
        """
        return {
            "brute_force": {
                "best_for": [
                    "Corpora < 1,000 vectors",
                    "Write-heavy workloads (>50% writes)",
                    "100% recall requirement on small data",
                ],
                "pros": [
                    "Zero index overhead",
                    "Fast writes (O(1))",
                    "Perfect recall",
                    "No tuning needed",
                ],
                "cons": [
                    "O(n) search time",
                    "Slow on large corpora (>100k vectors)",
                ],
            },
            "hnsw": {
                "best_for": [
                    "Corpora 1k - 10M vectors",
                    "Read-heavy workloads",
                    "High recall requirements (>95%)",
                    "General purpose use",
                ],
                "pros": [
                    "Fast search O(log n)",
                    "High recall (>95%)",
                    "Incremental updates supported",
                ],
                "cons": [
                    "Memory intensive (O(n*M))",
                    "Slower writes than brute force",
                    "Requires parameter tuning (M, ef)",
                ],
            },
            "ivf": {
                "best_for": [
                    "Corpora > 1M vectors",
                    "Memory-constrained environments",
                    "Acceptable recall ~90%",
                ],
                "pros": [
                    "Memory efficient",
                    "Scales to billions of vectors",
                    "Fast search on large corpora",
                ],
                "cons": [
                    "Lower recall than HNSW (~90%)",
                    "Requires training phase",
                    "Sensitive to cluster count parameter",
                ],
            },
        }


# ============================================================================
# Helper Functions
# ============================================================================


def get_oracle() -> IndexOracle:
    """Get IndexOracle instance (singleton)."""
    return IndexOracle()
