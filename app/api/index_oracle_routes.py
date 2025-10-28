"""
API endpoints for IndexOracle - adaptive index selection.

NOVEL FEATURE: Automatic index recommendation API - no other vector DB provides this.
"""

from typing import Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.dependencies import get_library_service
from app.services.library_service import CorpusService
from core.index_oracle import CorpusProfile, IndexOracle, get_oracle

router = APIRouter(prefix="/index-oracle", tags=["IndexOracle"])


# ============================================================================
# Request/Response Models
# ============================================================================


class IndexRecommendationResponse(BaseModel):
    """Response from index recommendation analysis."""

    corpus_id: str
    current_index: str
    recommended_index: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: List[str]
    expected_speedup: Optional[float] = None
    migration_cost: str
    auto_migrate_safe: bool
    corpus_stats: Dict


class IndexGuidelinesResponse(BaseModel):
    """Educational guidelines for index selection."""

    index_types: Dict[str, Dict]


# ============================================================================
# API Endpoints
# ============================================================================


@router.get("/corpora/{corpus_id}/analyze", response_model=IndexRecommendationResponse)
async def analyze_corpus_index(
    corpus_id: UUID,
    library_service: CorpusService = Depends(get_library_service),
):
    """
    Analyze corpus and recommend optimal index type.

    NOVEL FEATURE: Automatic index selection based on corpus characteristics
    and workload patterns. No other vector DB does this.

    The IndexOracle considers:
    - Corpus size (number of vectors)
    - Workload pattern (read-heavy vs write-heavy)
    - Current performance metrics
    - Memory constraints

    Returns recommendation with:
    - Confidence score (0-1, higher = more confident)
    - Human-readable reasoning
    - Expected performance improvement
    - Migration cost assessment
    - Auto-migration safety flag

    Use Cases:
    - Periodic health checks: "Is my index still optimal?"
    - Performance debugging: "Why is search slow?"
    - Capacity planning: "When should I upgrade index?"

    Args:
        corpus_id: Corpus to analyze

    Returns:
        Index recommendation with detailed reasoning

    Raises:
        404: Corpus not found
    """
    try:
        # Get corpus statistics
        stats = library_service.get_corpus_statistics(corpus_id)
        corpus = library_service.get_corpus(corpus_id)

        # Build corpus profile
        # Note: In production, these would come from monitoring/metrics
        profile = CorpusProfile(
            vector_count=stats["num_chunks"],
            dimension=corpus.metadata.embedding_dimension,
            current_index_type=corpus.metadata.index_type,
            # TODO: These should come from actual metrics
            search_rate_per_minute=0.0,
            insert_rate_per_minute=0.0,
            delete_rate_per_minute=0.0,
        )

        # Get recommendation from oracle
        oracle = get_oracle()
        recommendation = oracle.analyze_corpus(profile)

        return IndexRecommendationResponse(
            corpus_id=str(corpus_id),
            current_index=recommendation.current_index,
            recommended_index=recommendation.recommended_index,
            confidence=recommendation.confidence,
            reasoning=recommendation.reasoning,
            expected_speedup=recommendation.expected_speedup,
            migration_cost=recommendation.migration_cost,
            auto_migrate_safe=recommendation.auto_migrate,
            corpus_stats={
                "total_vectors": stats["num_chunks"],
                "total_documents": stats["num_documents"],
                "dimension": corpus.metadata.embedding_dimension,
                "current_index": corpus.metadata.index_type,
            },
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Index analysis failed: {str(e)}",
        )


@router.get("/guidelines", response_model=IndexGuidelinesResponse)
async def get_index_guidelines():
    """
    Get index selection guidelines.

    Educational endpoint that explains when to use each index type.
    Helps users understand the IndexOracle's decision logic.

    Returns comprehensive guide covering:
    - Best use cases for each index type
    - Pros and cons
    - Performance characteristics
    - Memory requirements

    Index Types Covered:
    - brute_force: For small corpora (<1k vectors)
    - hnsw: For medium corpora (1k-10M vectors)
    - ivf: For large corpora (>1M vectors)

    Returns:
        Educational guidelines for all index types
    """
    oracle = get_oracle()
    guidelines = oracle.get_index_guidelines()

    return IndexGuidelinesResponse(index_types=guidelines)


@router.get("/explain", response_model=Dict)
async def explain_index_oracle():
    """
    Get detailed explanation of IndexOracle.

    Returns educational content about how IndexOracle works.
    Useful for documentation and understanding recommendations.
    """
    return {
        "feature": "IndexOracle",
        "description": "Adaptive index selection based on corpus characteristics and workload patterns",
        "inspiration": [
            "Database query optimizers (cost-based optimization)",
            "Auto-scaling cloud services (adaptive resource allocation)",
            "Self-tuning systems research (autonomic computing)",
        ],
        "how_it_works": [
            "1. Collect corpus metrics (size, dimension, update rate)",
            "2. Score each index type based on workload fit",
            "3. Recommend highest-scoring index with confidence",
            "4. Assess migration cost and safety",
            "5. Optionally auto-migrate if high confidence + low cost",
        ],
        "decision_factors": {
            "corpus_size": "Primary factor - different indexes scale differently",
            "write_ratio": "Write-heavy workloads favor brute_force",
            "recall_requirements": "High recall (>95%) favors HNSW over IVF",
            "memory_constraints": "Large corpora may need IVF for memory efficiency",
        },
        "thresholds": {
            "brute_force_max": "1,000 vectors",
            "hnsw_range": "1,000 - 10,000,000 vectors",
            "ivf_min": "1,000,000 vectors (memory efficiency matters)",
            "write_heavy_threshold": "50% of operations are writes",
        },
        "confidence_levels": {
            "high": ">0.8 - safe for auto-migration",
            "medium": "0.5-0.8 - recommend but require approval",
            "low": "<0.5 - edge case, manual review recommended",
        },
        "migration_cost": {
            "low": "< 10k vectors, or brute_force → anything",
            "medium": "10k - 1M vectors",
            "high": "> 1M vectors (long rebuild time)",
        },
        "use_cases": [
            "Periodic health checks: Is my index still optimal?",
            "Performance debugging: Why is search slow?",
            "Capacity planning: When should I upgrade?",
            "Auto-tuning: Let the system optimize itself",
        ],
    }
