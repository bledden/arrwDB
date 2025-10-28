"""
API endpoints for QueryExpansion - automatic query rewriting.

NOVEL FEATURE: Query expansion API - no other vector DB provides this.
"""

from typing import Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.dependencies import get_library_service
from app.services.library_service import CorpusService
from core.query_expansion import ExpandedQuery, ExpansionResult, get_expander

router = APIRouter(prefix="/query-expansion", tags=["QueryExpansion"])


# ============================================================================
# Request/Response Models
# ============================================================================


class ExpandedQueryResponse(BaseModel):
    """Response model for expanded query."""

    text: str
    weight: float
    expansion_type: str
    confidence: float


class ExpansionResponse(BaseModel):
    """Response from query expansion."""

    original_query: str
    expanded_queries: List[ExpandedQueryResponse]
    num_expansions: int
    strategy: str


class SearchWithExpansionRequest(BaseModel):
    """Request for search with expansion."""

    query: str
    k: int = Field(10, ge=1, le=100)
    strategy: str = Field("balanced", regex="^(conservative|balanced|aggressive)$")
    fusion_method: str = Field("rrf", regex="^(rrf|weighted)$")


# ============================================================================
# API Endpoints
# ============================================================================


@router.post("/expand", response_model=ExpansionResponse)
async def expand_query(
    query: str = Query(..., min_length=1),
    strategy: str = Query("balanced", regex="^(conservative|balanced|aggressive)$"),
    max_expansions: int = Query(5, ge=1, le=20),
):
    """
    Expand a query into multiple semantic variations.

    NOVEL FEATURE: Automatic query expansion with synonym detection,
    specificity adjustment, and paraphrasing.

    Strategies:
    - Conservative: High-confidence expansions only
    - Balanced: Mix of synonyms and specificity
    - Aggressive: All possible expansions

    Args:
        query: Original query text
        strategy: Expansion strategy
        max_expansions: Maximum expansions to generate

    Returns:
        Expanded queries with weights
    """
    try:
        expander = get_expander()
        result = expander.expand_query(
            query=query,
            strategy=strategy,
            max_expansions=max_expansions,
        )

        expanded_queries_response = [
            ExpandedQueryResponse(
                text=exp.text,
                weight=exp.weight,
                expansion_type=exp.expansion_type,
                confidence=exp.confidence,
            )
            for exp in result.expanded_queries
        ]

        return ExpansionResponse(
            original_query=result.original_query,
            expanded_queries=expanded_queries_response,
            num_expansions=result.num_expansions,
            strategy=result.strategy,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query expansion failed: {str(e)}",
        )


@router.get("/explain", response_model=Dict)
async def explain_query_expansion():
    """
    Get detailed explanation of QueryExpansion.
    """
    return {
        "feature": "QueryExpansion",
        "description": "Automatic query rewriting for improved recall",
        "strategies": {
            "conservative": "High-confidence expansions only (precision-focused)",
            "balanced": "Mix of synonyms and specificity (default)",
            "aggressive": "All possible expansions (recall-focused)",
        },
        "expansion_types": {
            "original": "Original query (weight=1.0)",
            "synonym": "Direct synonym replacement",
            "specific": "More specific phrasing",
            "paraphrase": "Alternative phrasing",
        },
        "fusion_methods": {
            "rrf": "Reciprocal Rank Fusion (rank-based, robust)",
            "weighted": "Weighted score combination",
        },
        "use_cases": [
            "Synonym handling: 'car' finds 'automobile' documents",
            "Terminology variation: 'ML' finds 'machine learning'",
            "Recall improvement: Multiple queries find more results",
        ],
    }
