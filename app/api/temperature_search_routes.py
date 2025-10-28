"""
API endpoints for TemperatureSearch feature.

NOVEL FEATURE: Temperature-controlled vector search for exploration vs exploitation.
Inspired by LLM sampling - no other vector DB API provides this.
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.dependencies import get_library_service
from app.services.library_service import CorpusService
from core.temperature_search import TemperatureSearch, search_with_temperature

router = APIRouter(prefix="/temperature-search", tags=["TemperatureSearch"])


# ============================================================================
# Request/Response Models
# ============================================================================


class TemperatureSearchRequest(BaseModel):
    """Request for temperature-controlled search."""

    query_text: str = Field(..., description="Natural language query")
    k: int = Field(10, ge=1, le=1000, description="Number of results to return")
    temperature: float = Field(
        1.0,
        ge=0.0,
        le=5.0,
        description="Temperature parameter (0=greedy, higher=more exploration)",
    )
    candidate_multiplier: int = Field(
        5,
        ge=1,
        le=20,
        description="Fetch k*multiplier candidates before sampling (higher=more diversity)",
    )
    distance_threshold: Optional[float] = Field(
        None, description="Optional maximum distance threshold"
    )


class TemperatureSearchResult(BaseModel):
    """Single search result with temperature sampling."""

    chunk_id: str
    document_id: str
    text: str
    distance: float
    metadata: dict


class TemperatureSearchResponse(BaseModel):
    """Response from temperature-controlled search."""

    corpus_id: str
    query: str
    k: int
    temperature: float
    candidate_pool_size: int
    diversity_score: float
    results: List[TemperatureSearchResult]


class TemperatureRecommendationResponse(BaseModel):
    """Recommended temperature for a use case."""

    use_case: str
    recommended_temperature: float
    explanation: str


# ============================================================================
# API Endpoints
# ============================================================================


@router.post("/corpora/{corpus_id}/search", response_model=TemperatureSearchResponse)
async def search_with_temperature_control(
    corpus_id: UUID,
    request: TemperatureSearchRequest,
    library_service: CorpusService = Depends(get_library_service),
):
    """
    Search with temperature-controlled exploration.

    NOVEL FEATURE: Apply temperature to vector search for diversity control.

    Temperature controls exploration vs exploitation:
    - 0.0: Pure greedy (standard nearest neighbor search)
    - 1.0: Balanced (recommended for most use cases)
    - 2.0: High diversity (exploratory search)

    This is inspired by LLM sampling where temperature controls creativity.
    In vector search, it controls result diversity.

    Args:
        corpus_id: Corpus to search
        request: Search parameters including temperature

    Returns:
        Temperature-sampled search results with diversity metrics

    Use Cases:
        - Recommendations: Avoid filter bubbles (temperature=1.5)
        - Research: Explore adjacent topics (temperature=1.0-2.0)
        - Deduplication: Penalize similar results (temperature=0.5)
    """
    try:
        # Step 1: Perform standard search with larger k for candidate pool
        candidate_k = request.k * request.candidate_multiplier

        base_results = library_service.search_with_text(
            corpus_id=corpus_id,
            query_text=request.query_text,
            k=candidate_k,
            distance_threshold=request.distance_threshold,
        )

        if not base_results:
            return TemperatureSearchResponse(
                corpus_id=str(corpus_id),
                query=request.query_text,
                k=request.k,
                temperature=request.temperature,
                candidate_pool_size=0,
                diversity_score=0.0,
                results=[],
            )

        # Step 2: Convert to (chunk_id, distance) format
        candidates = [(chunk.id, distance) for chunk, distance in base_results]

        # Step 3: Apply temperature sampling
        sampled = search_with_temperature(
            base_results=candidates,
            k=request.k,
            temperature=request.temperature,
        )

        # Step 4: Calculate diversity score
        searcher = TemperatureSearch()
        diversity = searcher.compute_diversity_score(sampled)

        # Step 5: Build response with full chunk data
        # Need to get chunks from original results
        chunk_map = {chunk.id: (chunk, distance) for chunk, distance in base_results}

        results = []
        for chunk_id, distance in sampled:
            chunk, _ = chunk_map[chunk_id]
            results.append(
                TemperatureSearchResult(
                    chunk_id=str(chunk.id),
                    document_id=str(chunk.metadata.source_document_id),
                    text=chunk.text,
                    distance=distance,
                    metadata={
                        "chunk_index": chunk.metadata.chunk_index,
                        "created_at": chunk.metadata.created_at.isoformat(),
                    },
                )
            )

        return TemperatureSearchResponse(
            corpus_id=str(corpus_id),
            query=request.query_text,
            k=request.k,
            temperature=request.temperature,
            candidate_pool_size=len(candidates),
            diversity_score=diversity,
            results=results,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Temperature search failed: {str(e)}",
        )


@router.get("/recommend", response_model=TemperatureRecommendationResponse)
async def get_temperature_recommendation(
    use_case: str = Query(
        ...,
        description="Use case: 'precision', 'balanced', 'diversity', or 'exploration'",
    )
):
    """
    Get recommended temperature for common use cases.

    Helps users choose appropriate temperature without experimentation.

    Args:
        use_case: One of:
            - "precision": Best matches only (T=0.0)
            - "balanced": Mix of precision and diversity (T=1.0)
            - "diversity": High diversity (T=1.5)
            - "exploration": Maximum diversity (T=2.0)

    Returns:
        Recommended temperature and explanation
    """
    searcher = TemperatureSearch()

    try:
        temperature, explanation = searcher.recommend_temperature(use_case)

        return TemperatureRecommendationResponse(
            use_case=use_case,
            recommended_temperature=temperature,
            explanation=explanation,
        )
    except KeyError:
        # Invalid use case - return balanced as default
        temperature, explanation = searcher.recommend_temperature("balanced")

        return TemperatureRecommendationResponse(
            use_case="balanced",  # Override to valid use case
            recommended_temperature=temperature,
            explanation=f"Unknown use case '{use_case}'. Using 'balanced' as default. "
            + explanation,
        )


@router.get("/explain", response_model=dict)
async def explain_temperature_search():
    """
    Get detailed explanation of temperature-controlled search.

    Returns educational content about how temperature affects search behavior.
    Useful for documentation and user onboarding.
    """
    return {
        "feature": "TemperatureSearch",
        "description": "Temperature-controlled vector search for exploration vs exploitation",
        "inspiration": [
            "LLM sampling (GPT temperature parameter)",
            "Reinforcement Learning (exploration-exploitation trade-off)",
            "Information Retrieval (diversity vs relevance)",
        ],
        "how_it_works": [
            "1. Fetch candidate pool (k * candidate_multiplier)",
            "2. Convert distances to probabilities via softmax",
            "3. Apply temperature scaling (lower T = sharper distribution)",
            "4. Sample k results weighted by probability",
            "5. Return sampled results sorted by distance",
        ],
        "temperature_guide": {
            "0.0": {
                "name": "Pure Greedy",
                "behavior": "Returns exact top-k nearest neighbors",
                "use_when": "Exact matching, deduplication, high precision required",
            },
            "0.5": {
                "name": "Slight Smoothing",
                "behavior": "Prefers closer results but allows some variation",
                "use_when": "Mostly precision with minor diversity",
            },
            "1.0": {
                "name": "Balanced",
                "behavior": "Equal weight to precision and diversity",
                "use_when": "General search, recommendations, default use case",
            },
            "1.5": {
                "name": "High Diversity",
                "behavior": "Significant exploration beyond nearest neighbors",
                "use_when": "Recommendation diversity, discovering related content",
            },
            "2.0": {
                "name": "Maximum Exploration",
                "behavior": "Heavy exploration, less emphasis on proximity",
                "use_when": "Serendipitous discovery, research tools, avoiding filter bubbles",
            },
        },
        "use_cases": [
            "Recommendation engines: Avoid showing near-duplicates",
            "Research tools: Explore adjacent topics",
            "Content discovery: Find unexpected but relevant results",
            "A/B testing: Compare greedy vs exploratory retrieval",
        ],
        "metrics": {
            "diversity_score": "Variance of result distances (0-1, higher = more diverse)",
        },
        "parameters": {
            "temperature": "Controls exploration (0=greedy, higher=more random)",
            "candidate_multiplier": "Size of candidate pool (higher = more diversity but more compute)",
        },
    }
