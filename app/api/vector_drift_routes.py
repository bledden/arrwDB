"""
API endpoints for VectorDrift - distribution shift detection.

NOVEL FEATURE: Drift detection API - no other vector DB provides this.
"""

from typing import Dict, List
from uuid import UUID

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.dependencies import get_library_service
from app.services.library_service import CorpusService
from core.vector_drift import DriftStatistics, get_detector

router = APIRouter(prefix="/vector-drift", tags=["VectorDrift"])


# ============================================================================
# Request/Response Models
# ============================================================================


class DriftStatisticsResponse(BaseModel):
    """Response model for drift statistics."""

    mean_shift: float
    std_change: float
    ks_statistic: float
    ks_pvalue: float
    overlap_coefficient: float
    drift_severity: str


class DriftDetectionResponse(BaseModel):
    """Response from drift detection."""

    corpus_id: str
    baseline_id: str
    comparison_id: str
    baseline_size: int
    comparison_size: int
    statistics: DriftStatisticsResponse
    drift_detected: bool
    confidence: float
    recommendations: List[str]


# ============================================================================
# API Endpoints
# ============================================================================


@router.post("/corpora/{corpus_id}/detect", response_model=DriftDetectionResponse)
async def detect_drift(
    corpus_id: UUID,
    baseline_days: int = Query(30, ge=1, le=365),
    comparison_days: int = Query(7, ge=1, le=90),
    library_service: CorpusService = Depends(get_library_service),
):
    """
    Detect distribution drift in embeddings over time.

    NOVEL FEATURE: Monitors if embedding distribution has shifted,
    indicating model degradation or data changes.

    Args:
        corpus_id: Corpus to analyze
        baseline_days: Days to use as baseline period
        comparison_days: Recent days to compare

    Returns:
        Drift detection result with statistics
    """
    try:
        # Get corpus
        corpus = library_service.get_corpus(corpus_id)
        stats = library_service.get_corpus_statistics(corpus_id)

        if stats["num_chunks"] < 60:
            raise HTTPException(
                status_code=400,
                detail="Need at least 60 vectors for drift detection",
            )

        # Get embeddings
        vector_store = library_service._get_vector_store(corpus_id)
        total_vectors = vector_store.count

        # Simple split: first 70% = baseline, last 30% = comparison
        split_point = int(total_vectors * 0.7)

        baseline_embeddings = vector_store.vectors[:split_point]
        comparison_embeddings = vector_store.vectors[split_point:total_vectors]

        # Detect drift
        detector = get_detector()
        result = detector.detect_drift(
            corpus_id=corpus_id,
            baseline_embeddings=baseline_embeddings,
            comparison_embeddings=comparison_embeddings,
            baseline_id=f"first_{split_point}_vectors",
            comparison_id=f"last_{total_vectors-split_point}_vectors",
        )

        return DriftDetectionResponse(
            corpus_id=str(result.corpus_id),
            baseline_id=result.baseline_id,
            comparison_id=result.comparison_id,
            baseline_size=result.baseline_size,
            comparison_size=result.comparison_size,
            statistics=DriftStatisticsResponse(
                mean_shift=result.statistics.mean_shift,
                std_change=result.statistics.std_change,
                ks_statistic=result.statistics.ks_statistic,
                ks_pvalue=result.statistics.ks_pvalue,
                overlap_coefficient=result.statistics.overlap_coefficient,
                drift_severity=result.statistics.drift_severity,
            ),
            drift_detected=result.drift_detected,
            confidence=result.confidence,
            recommendations=result.recommendations,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Drift detection failed: {str(e)}",
        )


@router.get("/explain", response_model=Dict)
async def explain_drift_detection():
    """
    Get detailed explanation of VectorDrift.
    """
    return {
        "feature": "VectorDrift",
        "description": "Detect distribution shift in embeddings over time",
        "methods": {
            "ks_test": "Kolmogorov-Smirnov test for distribution differences",
            "mean_shift": "Euclidean distance between distribution means",
            "overlap": "Distribution overlap coefficient (0-1)",
        },
        "severity_levels": {
            "none": "No significant drift detected",
            "low": "Minor drift, monitoring recommended",
            "medium": "Moderate drift, investigation suggested",
            "high": "High drift, immediate action needed",
        },
        "use_cases": [
            "Model monitoring: Detect embedding model degradation",
            "A/B testing: Compare different model versions",
            "Data quality: Catch distribution shifts early",
        ],
    }
