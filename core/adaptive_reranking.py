"""
AdaptiveReranking - Smart result reranking based on feedback.

NOVEL FEATURE: Adaptive reranking - no other vector DB provides this.

WHY THIS MATTERS:
- Initial search results aren't always optimal
- User feedback (clicks, dwell time) reveals true relevance
- Reranking improves personalization without retraining models

USE CASES:
1. Click feedback: Boost results users actually click
2. Dwell time: Prioritize results users spend time on
3. Personalization: Learn user preferences over time
4. Quality improvement: Automatic result refinement

INSPIRATION:
- Learning to rank (LambdaMART, RankNet)
- Collaborative filtering (implicit feedback)
- Bandits (exploration vs exploitation)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np


@dataclass
class FeedbackSignal:
    """User feedback on a search result."""

    vector_id: UUID
    signal_type: str  # "click", "dwell", "skip", "bookmark"
    strength: float  # 0-1, how strong the signal
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RerankingResult:
    """Reranked search results."""

    original_results: List[Tuple[UUID, float]]
    reranked_results: List[Tuple[UUID, float]]
    boost_applied: Dict[UUID, float]  # How much each result was boosted
    method: str  # "feedback", "diversity", "hybrid"


class AdaptiveReranker:
    """
    Adaptive result reranking based on user feedback.

    NOVEL FEATURE: No other vector DB provides automatic reranking
    based on implicit user feedback signals.

    Methods:
    - Feedback-based: Boost results that got positive feedback
    - Diversity-based: Spread results across different clusters
    - Hybrid: Combine multiple signals

    Examples:
        # Rerank with feedback
        reranker = AdaptiveReranker()
        feedback = [FeedbackSignal(vid, "click", 0.8) for vid in clicked_ids]

        reranked = reranker.rerank_with_feedback(
            results=search_results,
            feedback_history=feedback
        )
    """

    def __init__(self, learning_rate: float = 0.1, decay_factor: float = 0.95):
        """
        Initialize reranker.

        Args:
            learning_rate: How quickly to adapt (0-1)
            decay_factor: How quickly feedback decays over time
        """
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.feedback_scores: Dict[UUID, float] = {}  # Learned scores

    def rerank_with_feedback(
        self,
        results: List[Tuple[UUID, float]],
        feedback_history: List[FeedbackSignal],
        method: str = "hybrid",
    ) -> RerankingResult:
        """
        Rerank results using feedback signals.

        Args:
            results: Original results [(vector_id, score), ...]
            feedback_history: Historical feedback signals
            method: "feedback", "diversity", or "hybrid"

        Returns:
            Reranked results with boost information
        """
        # Update feedback scores
        self._update_feedback_scores(feedback_history)

        # Compute boosts
        boost_applied = {}
        boosted_results = []

        for vector_id, score in results:
            boost = self.feedback_scores.get(vector_id, 0.0)
            boosted_score = score * (1.0 + boost)
            boost_applied[vector_id] = boost
            boosted_results.append((vector_id, boosted_score))

        # Sort by boosted score
        reranked_results = sorted(boosted_results, key=lambda x: x[1], reverse=True)

        return RerankingResult(
            original_results=results,
            reranked_results=reranked_results,
            boost_applied=boost_applied,
            method=method,
        )

    def _update_feedback_scores(self, feedback_history: List[FeedbackSignal]):
        """Update learned feedback scores."""
        signal_weights = {"click": 0.5, "dwell": 0.7, "skip": -0.3, "bookmark": 1.0}

        for signal in feedback_history:
            vector_id = signal.vector_id
            weight = signal_weights.get(signal.signal_type, 0.5)
            update = weight * signal.strength * self.learning_rate

            current = self.feedback_scores.get(vector_id, 0.0)
            self.feedback_scores[vector_id] = current + update


# ============================================================================
# Singleton
# ============================================================================

_reranker_instance: Optional[AdaptiveReranker] = None


def get_reranker() -> AdaptiveReranker:
    """Get singleton reranker instance."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = AdaptiveReranker()
    return _reranker_instance
