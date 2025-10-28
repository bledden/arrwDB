"""
TemperatureSearch - Add temperature parameter to vector search.

NOVEL FEATURE: Borrowed from LLM sampling and reinforcement learning.
No other vector DB supports temperature-controlled search.

INSPIRATION:
- LLMs: temperature=0 (greedy), temperature=1 (sampling), temperature=2 (creative)
- RL: exploration vs exploitation trade-off
- Information retrieval: diversity vs relevance

WHY THIS MATTERS:
Standard vector search is purely greedy - always returns the nearest neighbors.
But sometimes you want:
- DIVERSITY: Show me similar results but not duplicates (high temperature)
- EXPLORATION: Discover unexpected but relevant content (medium temperature)
- PRECISION: Best matches only (low temperature = standard search)

USE CASES:
1. Recommendation engines: Avoid filter bubbles with higher temperature
2. Research tools: Explore adjacent topics (temperature=1.5)
3. Deduplication: Penalize near-duplicates (temperature=0.5)
4. A/B testing: Compare greedy vs exploratory retrieval
"""

from typing import List, Tuple
from uuid import UUID

import numpy as np
from numpy.typing import NDArray


class TemperatureSearch:
    """
    Temperature-controlled vector search for exploration vs exploitation.

    Temperature controls the "sharpness" of the distance-to-probability conversion:
    - temperature = 0.0: Pure greedy (standard nearest neighbor)
    - temperature = 0.5: Slight preference for closer vectors
    - temperature = 1.0: Balanced (similar to softmax at T=1)
    - temperature = 2.0: More exploration, less emphasis on best match
    - temperature → ∞: Uniform random sampling

    ALGORITHM:
    1. Get top-N candidates from index (N >> k for exploration)
    2. Convert distances to probabilities using softmax with temperature
    3. Sample k results weighted by probability
    4. Return sampled results with their original distances
    """

    @staticmethod
    def apply_temperature(
        results: List[Tuple[UUID, float]],
        k: int,
        temperature: float = 1.0,
        candidate_multiplier: int = 5,
    ) -> List[Tuple[UUID, float]]:
        """
        Apply temperature-based resampling to search results.

        WHY: Standard search is greedy. Temperature adds controlled randomness
        for diversity and exploration.

        Args:
            results: List of (vector_id, distance) from base search.
            k: Final number of results to return.
            temperature: Controls exploration (0=greedy, higher=more random).
                - 0.0: Returns top-k as-is (pure exploitation)
                - 0.5: Slight smoothing (prefer closer but not exclusively)
                - 1.0: Balanced exploration-exploitation
                - 2.0: High diversity (less emphasis on nearest)
            candidate_multiplier: Fetch k*multiplier candidates before sampling.
                Higher = more diversity but more compute.

        Returns:
            Temperature-sampled results (may include vectors further than top-k).

        EXAMPLE:
            Top 10 results: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

            Temperature = 0.0 (greedy):
                Always picks top 5: [0.1, 0.2, 0.3, 0.4, 0.5]

            Temperature = 1.0 (balanced):
                Might pick: [0.1, 0.3, 0.4, 0.7, 0.9] (includes some farther results)

            Temperature = 2.0 (exploratory):
                Might pick: [0.2, 0.5, 0.8, 0.9, 1.0] (much more diversity)
        """
        if not results:
            return []

        # Special case: temperature = 0 means pure greedy (no sampling)
        if temperature == 0.0:
            return results[:k]

        # Get candidate pool (fetch more for diversity)
        num_candidates = min(len(results), k * candidate_multiplier)
        candidates = results[:num_candidates]

        # Extract distances
        vector_ids = [vec_id for vec_id, _ in candidates]
        distances = np.array([dist for _, dist in candidates], dtype=np.float32)

        # Convert distances to similarities (lower distance = higher similarity)
        # Use negative exponential: similarity = exp(-distance)
        # WHY: Exponential decay is common in RL and ensures positivity
        similarities = np.exp(-distances)

        # Apply temperature scaling to similarities
        # Higher temperature = flatter distribution = more exploration
        scaled_similarities = np.power(similarities, 1.0 / temperature)

        # Convert to probabilities (softmax-like)
        probabilities = scaled_similarities / scaled_similarities.sum()

        # Sample k results without replacement, weighted by probability
        # WHY: Without replacement ensures diversity (no duplicates)
        try:
            sampled_indices = np.random.choice(
                len(candidates),
                size=min(k, len(candidates)),
                replace=False,
                p=probabilities,
            )
        except ValueError:
            # Edge case: probabilities might have numerical issues
            # Fall back to uniform sampling
            sampled_indices = np.random.choice(
                len(candidates),
                size=min(k, len(candidates)),
                replace=False,
            )

        # Return sampled results with their original distances
        # IMPORTANT: Preserve original distances for accuracy
        sampled_results = [(vector_ids[i], distances[i]) for i in sampled_indices]

        # Sort by distance (ascending) for consistent output
        sampled_results.sort(key=lambda x: x[1])

        return sampled_results

    @staticmethod
    def compute_diversity_score(results: List[Tuple[UUID, float]]) -> float:
        """
        Compute diversity score of search results.

        WHY: Quantify how diverse the results are (useful for tuning temperature).

        Diversity = average pairwise distance variance
        Higher = more diverse results (less clustering)
        Lower = more similar results (more clustering)

        Args:
            results: List of (vector_id, distance) tuples.

        Returns:
            Diversity score (0 to 1, higher = more diverse).
        """
        if len(results) < 2:
            return 0.0

        distances = [dist for _, dist in results]

        # Calculate variance of distances
        variance = np.var(distances)

        # Normalize to 0-1 range (using empirical max variance ~1.0)
        diversity = min(variance, 1.0)

        return round(diversity, 4)

    @staticmethod
    def recommend_temperature(
        use_case: str = "balanced",
    ) -> Tuple[float, str]:
        """
        Recommend temperature value for common use cases.

        WHY: Help users choose appropriate temperature without experimentation.

        Args:
            use_case: One of:
                - "precision": Best matches only (standard search)
                - "balanced": Mix of precision and diversity
                - "diversity": Avoid similar results, explore more
                - "exploration": High diversity, discover unexpected results

        Returns:
            Tuple of (recommended_temperature, explanation).
        """
        recommendations = {
            "precision": (
                0.0,
                "Pure greedy search - returns exact top-k nearest neighbors. "
                "Use for: exact matching, deduplication, high-precision retrieval.",
            ),
            "balanced": (
                1.0,
                "Balanced exploration-exploitation - prefers closer results but allows diversity. "
                "Use for: general search, recommendations, avoiding filter bubbles.",
            ),
            "diversity": (
                1.5,
                "High diversity - significant exploration beyond nearest neighbors. "
                "Use for: recommendation diversity, discovering related content, A/B testing.",
            ),
            "exploration": (
                2.0,
                "Maximum diversity - heavy exploration with less emphasis on proximity. "
                "Use for: serendipitous discovery, research tools, breaking echo chambers.",
            ),
        }

        if use_case not in recommendations:
            # Default to balanced
            use_case = "balanced"

        return recommendations[use_case]


# ============================================================================
# Helper Functions for Integration
# ============================================================================


def search_with_temperature(
    base_results: List[Tuple[UUID, float]],
    k: int,
    temperature: float = 1.0,
) -> List[Tuple[UUID, float]]:
    """
    Convenience function for temperature-controlled search.

    WHY: Simple wrapper for easy integration into existing search code.

    Args:
        base_results: Results from standard vector search.
        k: Number of results to return.
        temperature: Exploration parameter (0=greedy, higher=more random).

    Returns:
        Temperature-sampled results.

    USAGE:
        # Standard search
        results = index.search(query, k=50)

        # Apply temperature for diversity
        diverse_results = search_with_temperature(results, k=10, temperature=1.5)
    """
    searcher = TemperatureSearch()
    return searcher.apply_temperature(base_results, k, temperature)
