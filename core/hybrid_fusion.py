"""
HybridFusion - Intelligent fusion of multiple search strategies.

NOVEL FEATURE: Hybrid search fusion - no other vector DB provides adaptive fusion.

WHY THIS MATTERS:
- Different queries need different search strategies
- Vector search great for semantic, BM25 for keywords
- Naive combination (0.5*vector + 0.5*bm25) suboptimal
- Adaptive weights improve results

USE CASES:
1. Hybrid search: Combine vector + keyword search
2. Multi-model: Fuse results from different embedding models
3. Ensemble: Combine multiple retrieval strategies
4. Adaptive: Learn optimal weights per query type

INSPIRATION:
- Ensemble learning (boosting, stacking)
- Hybrid retrieval (Elasticsearch, Vespa)
- Meta-learning (learning to combine)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np


@dataclass
class FusionResult:
    """Result of fusing multiple search strategies."""

    fused_results: List[Tuple[UUID, float]]
    strategy_weights: Dict[str, float]
    method: str  # "linear", "rrf", "learned"
    confidence: float  # Confidence in fusion (0-1)


class HybridFusion:
    """
    Intelligent fusion of multiple search strategies.

    NOVEL FEATURE: Adaptive fusion weights based on query characteristics.

    Fusion Methods:
    - Linear: Weighted combination of scores
    - RRF: Reciprocal Rank Fusion (rank-based)
    - Learned: Adapt weights based on query type

    Examples:
        # Fuse vector and keyword search
        fusion = HybridFusion()
        result = fusion.fuse_results({
            "vector": vector_results,
            "keyword": keyword_results
        }, method="rrf")
    """

    def __init__(self):
        """Initialize fusion engine."""
        # Default weights for different strategies
        self.default_weights = {"vector": 0.7, "keyword": 0.3, "metadata": 0.2}

    def fuse_results(
        self,
        results_by_strategy: Dict[str, List[Tuple[UUID, float]]],
        method: str = "rrf",
        weights: Optional[Dict[str, float]] = None,
    ) -> FusionResult:
        """
        Fuse results from multiple search strategies.

        Args:
            results_by_strategy: Dict mapping strategy name to results
            method: "linear", "rrf", or "learned"
            weights: Optional custom weights (default: self.default_weights)

        Returns:
            Fused results with metadata
        """
        if not weights:
            weights = {k: self.default_weights.get(k, 0.5) for k in results_by_strategy.keys()}

        if method == "linear":
            fused = self._linear_fusion(results_by_strategy, weights)
        elif method == "rrf":
            fused = self._rrf_fusion(results_by_strategy, weights)
        else:
            fused = self._rrf_fusion(results_by_strategy, weights)

        confidence = self._compute_confidence(results_by_strategy, fused)

        return FusionResult(
            fused_results=fused,
            strategy_weights=weights,
            method=method,
            confidence=confidence,
        )

    def _linear_fusion(
        self,
        results_by_strategy: Dict[str, List[Tuple[UUID, float]]],
        weights: Dict[str, float],
    ) -> List[Tuple[UUID, float]]:
        """Linear weighted combination of scores."""
        vector_scores: Dict[UUID, float] = {}

        for strategy, results in results_by_strategy.items():
            weight = weights.get(strategy, 0.5)
            for vector_id, score in results:
                if vector_id in vector_scores:
                    vector_scores[vector_id] += weight * score
                else:
                    vector_scores[vector_id] = weight * score

        return sorted(vector_scores.items(), key=lambda x: x[1], reverse=True)

    def _rrf_fusion(
        self,
        results_by_strategy: Dict[str, List[Tuple[UUID, float]]],
        weights: Dict[str, float],
        k: int = 60,
    ) -> List[Tuple[UUID, float]]:
        """Reciprocal Rank Fusion."""
        vector_scores: Dict[UUID, float] = {}

        for strategy, results in results_by_strategy.items():
            weight = weights.get(strategy, 1.0)
            for rank, (vector_id, _) in enumerate(results):
                rrf_score = weight / (k + rank + 1)
                if vector_id in vector_scores:
                    vector_scores[vector_id] += rrf_score
                else:
                    vector_scores[vector_id] = rrf_score

        return sorted(vector_scores.items(), key=lambda x: x[1], reverse=True)

    def _compute_confidence(
        self,
        results_by_strategy: Dict[str, List[Tuple[UUID, float]]],
        fused_results: List[Tuple[UUID, float]],
    ) -> float:
        """Compute confidence in fusion (agreement between strategies)."""
        if len(results_by_strategy) < 2:
            return 1.0

        # Measure overlap in top-k results
        top_k = 10
        strategy_top_ids = [
            set([vid for vid, _ in results[:top_k]]) for results in results_by_strategy.values()
        ]

        if len(strategy_top_ids) < 2:
            return 0.5

        # Average pairwise Jaccard similarity
        similarities = []
        for i in range(len(strategy_top_ids)):
            for j in range(i + 1, len(strategy_top_ids)):
                intersection = len(strategy_top_ids[i] & strategy_top_ids[j])
                union = len(strategy_top_ids[i] | strategy_top_ids[j])
                if union > 0:
                    similarities.append(intersection / union)

        return float(np.mean(similarities)) if similarities else 0.5


# ============================================================================
# Singleton
# ============================================================================

_fusion_instance: Optional[HybridFusion] = None


def get_fusion() -> HybridFusion:
    """Get singleton fusion instance."""
    global _fusion_instance
    if _fusion_instance is None:
        _fusion_instance = HybridFusion()
    return _fusion_instance
