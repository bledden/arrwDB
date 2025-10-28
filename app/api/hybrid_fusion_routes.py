"""
API endpoints for HybridFusion - intelligent search fusion.

NOVEL FEATURE: Hybrid fusion API - no other vector DB provides adaptive fusion.
"""

from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.hybrid_fusion import get_fusion

router = APIRouter(prefix="/hybrid-fusion", tags=["HybridFusion"])


# ============================================================================
# Request/Response Models
# ============================================================================


class FusionRequest(BaseModel):
    """Request for fusion."""

    results_by_strategy: Dict[str, List[Dict[str, float]]]
    method: str = Field("rrf", regex="^(linear|rrf|learned)$")
    weights: Optional[Dict[str, float]] = None


class FusionResponse(BaseModel):
    """Response from fusion."""

    fused_results: List[Dict[str, float]]
    strategy_weights: Dict[str, float]
    method: str
    confidence: float


# ============================================================================
# API Endpoints
# ============================================================================


@router.post("/fuse", response_model=FusionResponse)
async def fuse_results(request: FusionRequest):
    """
    Fuse results from multiple search strategies.

    NOVEL FEATURE: Intelligent fusion with adaptive weights.

    Args:
        request: Fusion request with results from different strategies

    Returns:
        Fused results with confidence score
    """
    try:
        # Parse results
        from uuid import UUID

        results_by_strategy = {}
        for strategy, results in request.results_by_strategy.items():
            results_by_strategy[strategy] = [
                (UUID(r["vector_id"]), r["score"]) for r in results
            ]

        # Fuse
        fusion = get_fusion()
        result = fusion.fuse_results(
            results_by_strategy=results_by_strategy,
            method=request.method,
            weights=request.weights,
        )

        return FusionResponse(
            fused_results=[
                {"vector_id": str(vid), "score": score}
                for vid, score in result.fused_results
            ],
            strategy_weights=result.strategy_weights,
            method=result.method,
            confidence=result.confidence,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Fusion failed: {str(e)}",
        )


@router.get("/explain", response_model=Dict)
async def explain_hybrid_fusion():
    """
    Get detailed explanation of HybridFusion.
    """
    return {
        "feature": "HybridFusion",
        "description": "Intelligent fusion of multiple search strategies",
        "methods": {
            "linear": "Weighted combination of scores",
            "rrf": "Reciprocal Rank Fusion (rank-based)",
            "learned": "Adaptive weights based on query type",
        },
        "default_weights": {
            "vector": 0.7,
            "keyword": 0.3,
            "metadata": 0.2,
        },
        "use_cases": [
            "Hybrid search: Combine vector + keyword search",
            "Multi-model: Fuse results from different embeddings",
            "Ensemble: Combine multiple retrieval strategies",
        ],
    }
