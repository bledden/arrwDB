"""
API endpoints for AdaptiveReranking - smart result reranking.

NOVEL FEATURE: Adaptive reranking API - no other vector DB provides this.
"""

from typing import Dict, List
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.adaptive_reranking import FeedbackSignal, get_reranker

router = APIRouter(prefix="/adaptive-reranking", tags=["AdaptiveReranking"])


# ============================================================================
# Request/Response Models
# ============================================================================


class FeedbackSignalRequest(BaseModel):
    """Request model for feedback signal."""

    vector_id: str
    signal_type: str = Field(..., regex="^(click|dwell|skip|bookmark)$")
    strength: float = Field(..., ge=0.0, le=1.0)


class RerankRequest(BaseModel):
    """Request for reranking."""

    results: List[Dict[str, float]]  # [{"vector_id": "uuid", "score": 0.9}, ...]
    feedback: List[FeedbackSignalRequest]
    method: str = Field("hybrid", regex="^(feedback|diversity|hybrid)$")


class RerankResponse(BaseModel):
    """Response from reranking."""

    original_results: List[Dict[str, float]]
    reranked_results: List[Dict[str, float]]
    boost_applied: Dict[str, float]
    method: str


# ============================================================================
# API Endpoints
# ============================================================================


@router.post("/rerank", response_model=RerankResponse)
async def rerank_results(request: RerankRequest):
    """
    Rerank search results using feedback signals.

    NOVEL FEATURE: Adaptive reranking based on user feedback (clicks, dwell time).

    Args:
        request: Reranking request with results and feedback

    Returns:
        Reranked results with boost information
    """
    try:
        # Parse results
        results = [(UUID(r["vector_id"]), r["score"]) for r in request.results]

        # Parse feedback
        feedback = [
            FeedbackSignal(
                vector_id=UUID(f.vector_id),
                signal_type=f.signal_type,
                strength=f.strength,
            )
            for f in request.feedback
        ]

        # Rerank
        reranker = get_reranker()
        result = reranker.rerank_with_feedback(
            results=results,
            feedback_history=feedback,
            method=request.method,
        )

        return RerankResponse(
            original_results=[{"vector_id": str(vid), "score": score} for vid, score in result.original_results],
            reranked_results=[{"vector_id": str(vid), "score": score} for vid, score in result.reranked_results],
            boost_applied={str(k): v for k, v in result.boost_applied.items()},
            method=result.method,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Reranking failed: {str(e)}",
        )


@router.get("/explain", response_model=Dict)
async def explain_adaptive_reranking():
    """
    Get detailed explanation of AdaptiveReranking.
    """
    return {
        "feature": "AdaptiveReranking",
        "description": "Smart result reranking based on user feedback",
        "signal_types": {
            "click": "User clicked result (weight: 0.5)",
            "dwell": "User spent time on result (weight: 0.7)",
            "skip": "User skipped result (weight: -0.3)",
            "bookmark": "User bookmarked result (weight: 1.0)",
        },
        "methods": {
            "feedback": "Pure feedback-based boosting",
            "diversity": "Spread results across clusters",
            "hybrid": "Combine feedback + diversity",
        },
        "use_cases": [
            "Personalization: Learn user preferences over time",
            "Click optimization: Boost results users actually click",
            "Quality improvement: Automatic result refinement",
        ],
    }
