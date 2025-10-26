"""
Hybrid search combining vector similarity with metadata-based scoring.

This module provides advanced query features:
- Hybrid search (vector + metadata scoring)
- Query-time field boosting
- Custom reranking functions
- Pre-filtering optimization
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

from app.models.base import Chunk

logger = logging.getLogger(__name__)


@dataclass
class ScoringConfig:
    """Configuration for hybrid search scoring."""

    # Weight for vector similarity score (0.0 to 1.0)
    vector_weight: float = 0.7

    # Weight for metadata boost score (0.0 to 1.0)
    # Note: vector_weight + metadata_weight should equal 1.0
    metadata_weight: float = 0.3

    # Field-specific boost factors
    # Example: {"tags": 2.0, "author": 1.5} boosts matching tags/authors
    field_boosts: Optional[Dict[str, float]] = None

    # Recency boost (boost newer documents)
    recency_boost_enabled: bool = False
    recency_half_life_days: float = 30.0  # Documents lose half boost after N days

    # Normalize scores to 0-1 range
    normalize_scores: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.vector_weight < 0 or self.vector_weight > 1:
            raise ValueError("vector_weight must be between 0 and 1")
        if self.metadata_weight < 0 or self.metadata_weight > 1:
            raise ValueError("metadata_weight must be between 0 and 1")
        if abs((self.vector_weight + self.metadata_weight) - 1.0) > 0.01:
            raise ValueError("vector_weight + metadata_weight must equal 1.0")


class HybridSearchScorer:
    """
    Hybrid search scorer that combines vector similarity with metadata signals.

    This class implements production-grade ranking similar to:
    - Elasticsearch's "function_score" queries
    - Pinecone's "metadata_filter" with score boosting
    - Weaviate's "hybrid search" mode
    """

    def __init__(self, config: Optional[ScoringConfig] = None):
        """
        Initialize hybrid search scorer.

        Args:
            config: Scoring configuration. Defaults to balanced config (0.7 vector, 0.3 metadata).
        """
        self.config = config or ScoringConfig()
        logger.info(
            f"HybridSearchScorer initialized: "
            f"vector_weight={self.config.vector_weight}, "
            f"metadata_weight={self.config.metadata_weight}"
        )

    def score_results(
        self,
        results: List[Tuple[Chunk, float]],
        query_metadata: Optional[Dict] = None,
    ) -> List[Tuple[Chunk, float, Dict]]:
        """
        Apply hybrid scoring to search results.

        Args:
            results: List of (chunk, distance) tuples from vector search
            query_metadata: Optional metadata to match against for boosting

        Returns:
            List of (chunk, hybrid_score, score_breakdown) tuples sorted by hybrid score

        The hybrid score combines:
        1. Vector similarity (cosine distance converted to similarity)
        2. Metadata boost (field matches, recency, etc.)
        """
        if not results:
            return []

        logger.info(f"Scoring {len(results)} results with hybrid search")

        scored_results = []
        for chunk, distance in results:
            # Convert distance to similarity score (0-1, where 1 is most similar)
            # Cosine distance is in range [0, 2], so similarity = 1 - (distance / 2)
            vector_score = 1.0 - (distance / 2.0)
            vector_score = max(0.0, min(1.0, vector_score))  # Clamp to [0, 1]

            # Calculate metadata boost score
            metadata_score, score_breakdown = self._calculate_metadata_score(
                chunk, query_metadata
            )

            # Combine scores with weights
            hybrid_score = (
                self.config.vector_weight * vector_score
                + self.config.metadata_weight * metadata_score
            )

            # Build detailed breakdown
            breakdown = {
                "vector_score": vector_score,
                "vector_distance": distance,
                "metadata_score": metadata_score,
                "hybrid_score": hybrid_score,
                "vector_weight": self.config.vector_weight,
                "metadata_weight": self.config.metadata_weight,
                **score_breakdown,
            }

            scored_results.append((chunk, hybrid_score, breakdown))

        # Sort by hybrid score (descending)
        scored_results.sort(key=lambda x: x[1], reverse=True)

        logger.info(
            f"Hybrid scoring complete: top score={scored_results[0][1]:.4f}, "
            f"bottom score={scored_results[-1][1]:.4f}"
        )

        return scored_results

    def _calculate_metadata_score(
        self, chunk: Chunk, query_metadata: Optional[Dict]
    ) -> Tuple[float, Dict]:
        """
        Calculate metadata-based boost score for a chunk.

        Returns:
            (score, breakdown) where score is in [0, 1] and breakdown contains details
        """
        score = 0.0
        breakdown = {}

        # Field boost scoring
        if self.config.field_boosts and query_metadata:
            field_score, field_breakdown = self._calculate_field_boosts(
                chunk, query_metadata
            )
            score += field_score
            breakdown["field_boost"] = field_breakdown

        # Recency boost scoring
        if self.config.recency_boost_enabled:
            recency_score = self._calculate_recency_boost(chunk)
            score += recency_score
            breakdown["recency_boost"] = recency_score

        # Normalize to [0, 1] if multiple boosts are enabled
        if self.config.normalize_scores:
            num_boosts = sum(
                [
                    bool(self.config.field_boosts and query_metadata),
                    bool(self.config.recency_boost_enabled),
                ]
            )
            if num_boosts > 0:
                score = score / num_boosts

        return min(1.0, score), breakdown

    def _calculate_field_boosts(
        self, chunk: Chunk, query_metadata: Dict
    ) -> Tuple[float, Dict]:
        """
        Calculate boost based on field matches.

        Example:
        - If query specifies author="Alice" and chunk's document has author="Alice",
          boost score by field_boosts["author"]
        - If query specifies tags=["AI"] and chunk's document has tags=["AI", "ML"],
          boost score by field_boosts["tags"]
        """
        if not self.config.field_boosts:
            return 0.0, {}

        score = 0.0
        breakdown = {}

        # Get document metadata from chunk
        # Note: We need to enhance this to access document metadata
        # For now, we'll work with what we have in chunk metadata

        for field, boost_factor in self.config.field_boosts.items():
            if field in query_metadata:
                query_value = query_metadata[field]

                # Check if chunk/document has matching field
                # This is a simplified implementation - in production you'd
                # want to join with document metadata
                match_score = 0.0

                if field == "tags":
                    # Tag matching logic
                    # For now, return 0 as we need document-level metadata
                    match_score = 0.0
                elif field == "author":
                    # Author matching logic
                    match_score = 0.0

                if match_score > 0:
                    score += boost_factor * match_score
                    breakdown[field] = boost_factor * match_score

        # Normalize by total boost factors
        total_boost = sum(self.config.field_boosts.values())
        if total_boost > 0:
            score = score / total_boost

        return min(1.0, score), breakdown

    def _calculate_recency_boost(self, chunk: Chunk) -> float:
        """
        Calculate recency boost using exponential decay.

        Formula: score = 2^(-age_days / half_life_days)

        This gives:
        - score = 1.0 for documents created today
        - score = 0.5 for documents at half-life age
        - score = 0.25 for documents at 2x half-life age
        - etc.
        """
        try:
            created_at = chunk.metadata.created_at
            age_days = (datetime.utcnow() - created_at).days

            # Exponential decay
            score = 2.0 ** (-age_days / self.config.recency_half_life_days)

            return min(1.0, score)
        except Exception as e:
            logger.warning(f"Failed to calculate recency boost: {e}")
            return 0.0


class ResultReranker:
    """
    Rerank search results using custom scoring functions.

    This allows post-processing of search results with arbitrary logic,
    similar to Elasticsearch's "rescore" feature.
    """

    def __init__(self, scoring_fn: Callable[[Chunk, float], float]):
        """
        Initialize reranker with custom scoring function.

        Args:
            scoring_fn: Function that takes (chunk, original_score) and returns new score
        """
        self.scoring_fn = scoring_fn

    def rerank(
        self, results: List[Tuple[Chunk, float]]
    ) -> List[Tuple[Chunk, float]]:
        """
        Rerank results using custom scoring function.

        Args:
            results: List of (chunk, score) tuples

        Returns:
            Reranked list of (chunk, score) tuples
        """
        logger.info(f"Reranking {len(results)} results")

        reranked = []
        for chunk, score in results:
            new_score = self.scoring_fn(chunk, score)
            reranked.append((chunk, new_score))

        # Sort by new score (descending)
        reranked.sort(key=lambda x: x[1], reverse=True)

        return reranked


# Pre-built reranking functions


def boost_by_recency(
    half_life_days: float = 30.0,
) -> Callable[[Chunk, float], float]:
    """
    Create a reranking function that boosts recent documents.

    Args:
        half_life_days: Documents lose half their boost after this many days

    Returns:
        Reranking function
    """

    def rerank_fn(chunk: Chunk, original_score: float) -> float:
        try:
            created_at = chunk.metadata.created_at
            age_days = (datetime.utcnow() - created_at).days

            # Exponential decay boost
            recency_boost = 2.0 ** (-age_days / half_life_days)

            # Combine original score with recency (70% original, 30% recency)
            return 0.7 * original_score + 0.3 * recency_boost
        except Exception:
            return original_score

    return rerank_fn


def boost_by_chunk_position(
    prefer_early: bool = True,
) -> Callable[[Chunk, float], float]:
    """
    Create a reranking function that boosts chunks by position in document.

    Args:
        prefer_early: If True, boost earlier chunks. If False, boost later chunks.

    Returns:
        Reranking function
    """

    def rerank_fn(chunk: Chunk, original_score: float) -> float:
        try:
            chunk_index = chunk.metadata.chunk_index

            # Normalize chunk index (assume max 100 chunks per doc)
            normalized_position = min(1.0, chunk_index / 100.0)

            if prefer_early:
                # Boost earlier chunks (position 0 = max boost)
                position_boost = 1.0 - normalized_position
            else:
                # Boost later chunks (position 1 = max boost)
                position_boost = normalized_position

            # Combine original score with position (80% original, 20% position)
            return 0.8 * original_score + 0.2 * position_boost
        except Exception:
            return original_score

    return rerank_fn


def boost_by_length(
    prefer_longer: bool = True,
) -> Callable[[Chunk, float], float]:
    """
    Create a reranking function that boosts chunks by text length.

    Args:
        prefer_longer: If True, boost longer chunks. If False, boost shorter chunks.

    Returns:
        Reranking function
    """

    def rerank_fn(chunk: Chunk, original_score: float) -> float:
        try:
            text_length = len(chunk.text)

            # Normalize length (assume max 5000 chars per chunk)
            normalized_length = min(1.0, text_length / 5000.0)

            if prefer_longer:
                length_boost = normalized_length
            else:
                length_boost = 1.0 - normalized_length

            # Combine original score with length (85% original, 15% length)
            return 0.85 * original_score + 0.15 * length_boost
        except Exception:
            return original_score

    return rerank_fn
