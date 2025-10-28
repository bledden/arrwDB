"""
VectorDrift - Detect distribution shift in embeddings over time.

NOVEL FEATURE: Embedding drift detection - no other vector DB provides this.

WHY THIS MATTERS:
- ML models degrade over time (concept drift)
- Embedding models can change (version updates)
- Data distribution shifts (new topics emerge)
- Early detection prevents quality degradation

USE CASES:
1. Model monitoring: "Has my embedding model degraded?"
2. Data quality: "Are new embeddings different from old ones?"
3. Concept drift: "Is my data distribution changing?"
4. A/B testing: "Does the new model produce different embeddings?"

INSPIRATION:
- ML monitoring (Evidently AI, Fiddler)
- Statistical hypothesis testing (KS test, chi-square)
- Distribution shift detection research
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple
from uuid import UUID

import numpy as np
from numpy.typing import NDArray


@dataclass
class DriftStatistics:
    """Statistical comparison between two embedding distributions."""

    # Basic statistics
    mean_shift: float  # Euclidean distance between means
    std_change: float  # Ratio of standard deviations
    dimension_shifts: List[float]  # Per-dimension mean shifts

    # Distribution tests
    ks_statistic: float  # Kolmogorov-Smirnov test (0-1, higher = more drift)
    ks_pvalue: float  # P-value (< 0.05 = significant drift)

    # Practical metrics
    overlap_coefficient: float  # Distribution overlap (0-1, higher = more similar)
    drift_severity: str  # "none", "low", "medium", "high"


@dataclass
class DriftDetectionResult:
    """Complete drift detection result."""

    corpus_id: UUID
    baseline_id: str  # Identifier for baseline period
    comparison_id: str  # Identifier for comparison period

    baseline_size: int
    comparison_size: int

    statistics: DriftStatistics
    drift_detected: bool
    confidence: float  # Confidence in drift detection (0-1)

    timestamp: datetime = field(default_factory=datetime.utcnow)
    recommendations: List[str] = field(default_factory=list)


class VectorDriftDetector:
    """
    Detect distribution shift in embeddings over time.

    NOVEL FEATURE: No other vector DB monitors embedding drift.
    This helps catch model degradation, data shifts, and quality issues early.

    Detection Methods:
    - Statistical tests (Kolmogorov-Smirnov)
    - Mean/std shift analysis
    - Per-dimension drift detection
    - Distribution overlap coefficient

    Examples:
        # Compare two time periods
        detector = VectorDriftDetector()
        result = detector.detect_drift(
            corpus_id=corpus_id,
            baseline_embeddings=old_embeddings,
            comparison_embeddings=new_embeddings
        )

        if result.drift_detected:
            print(f"Drift severity: {result.statistics.drift_severity}")
            print(f"Recommendations: {result.recommendations}")
    """

    def __init__(
        self,
        significance_level: float = 0.05,  # P-value threshold
        min_sample_size: int = 30,  # Minimum samples per period
    ):
        """
        Initialize drift detector.

        Args:
            significance_level: P-value threshold for statistical tests
            min_sample_size: Minimum samples needed for detection
        """
        self.significance_level = significance_level
        self.min_sample_size = min_sample_size

    def detect_drift(
        self,
        corpus_id: UUID,
        baseline_embeddings: NDArray[np.float32],
        comparison_embeddings: NDArray[np.float32],
        baseline_id: str = "baseline",
        comparison_id: str = "current",
    ) -> DriftDetectionResult:
        """
        Detect drift between two embedding distributions.

        Args:
            corpus_id: Corpus identifier
            baseline_embeddings: Historical embeddings (n1, dim)
            comparison_embeddings: Recent embeddings (n2, dim)
            baseline_id: Label for baseline period
            comparison_id: Label for comparison period

        Returns:
            Drift detection result with statistics

        Raises:
            ValueError: If embeddings are invalid or too small
        """
        # Validation
        if len(baseline_embeddings) < self.min_sample_size:
            raise ValueError(
                f"Baseline too small: {len(baseline_embeddings)} < {self.min_sample_size}"
            )
        if len(comparison_embeddings) < self.min_sample_size:
            raise ValueError(
                f"Comparison too small: {len(comparison_embeddings)} < {self.min_sample_size}"
            )

        # Compute statistics
        statistics = self._compute_drift_statistics(baseline_embeddings, comparison_embeddings)

        # Determine if drift occurred
        drift_detected, confidence = self._classify_drift(statistics)

        # Generate recommendations
        recommendations = self._generate_recommendations(statistics, drift_detected)

        return DriftDetectionResult(
            corpus_id=corpus_id,
            baseline_id=baseline_id,
            comparison_id=comparison_id,
            baseline_size=len(baseline_embeddings),
            comparison_size=len(comparison_embeddings),
            statistics=statistics,
            drift_detected=drift_detected,
            confidence=confidence,
            recommendations=recommendations,
        )

    def _compute_drift_statistics(
        self,
        baseline: NDArray[np.float32],
        comparison: NDArray[np.float32],
    ) -> DriftStatistics:
        """Compute comprehensive drift statistics."""
        # Mean shift (Euclidean distance between means)
        baseline_mean = np.mean(baseline, axis=0)
        comparison_mean = np.mean(comparison, axis=0)
        mean_shift = float(np.linalg.norm(baseline_mean - comparison_mean))

        # Standard deviation change (ratio)
        baseline_std = float(np.std(baseline))
        comparison_std = float(np.std(comparison))
        std_change = comparison_std / (baseline_std + 1e-8)

        # Per-dimension shifts
        dimension_shifts = np.abs(baseline_mean - comparison_mean).tolist()

        # Kolmogorov-Smirnov test (univariate, use first principal component)
        # For multivariate data, project onto line between means
        direction = comparison_mean - baseline_mean
        direction = direction / (np.linalg.norm(direction) + 1e-8)

        baseline_proj = np.dot(baseline, direction)
        comparison_proj = np.dot(comparison, direction)

        ks_stat, ks_pval = self._ks_test(baseline_proj, comparison_proj)

        # Distribution overlap coefficient
        overlap = self._compute_overlap(baseline, comparison)

        # Classify severity
        severity = self._classify_severity(mean_shift, ks_stat, overlap)

        return DriftStatistics(
            mean_shift=mean_shift,
            std_change=std_change,
            dimension_shifts=dimension_shifts,
            ks_statistic=ks_stat,
            ks_pvalue=ks_pval,
            overlap_coefficient=overlap,
            drift_severity=severity,
        )

    def _ks_test(
        self, baseline: NDArray[np.float32], comparison: NDArray[np.float32]
    ) -> Tuple[float, float]:
        """
        Kolmogorov-Smirnov two-sample test.

        WHY: Non-parametric test for distribution differences.
        Tests null hypothesis: both samples from same distribution.
        """
        # Sort both samples
        baseline_sorted = np.sort(baseline)
        comparison_sorted = np.sort(comparison)

        n1 = len(baseline_sorted)
        n2 = len(comparison_sorted)

        # Compute empirical CDFs
        all_values = np.concatenate([baseline_sorted, comparison_sorted])
        all_values = np.sort(np.unique(all_values))

        cdf1 = np.searchsorted(baseline_sorted, all_values, side="right") / n1
        cdf2 = np.searchsorted(comparison_sorted, all_values, side="right") / n2

        # KS statistic: max absolute difference between CDFs
        ks_statistic = float(np.max(np.abs(cdf1 - cdf2)))

        # Approximate p-value (asymptotic formula)
        en = np.sqrt(n1 * n2 / (n1 + n2))
        try:
            p_value = 2 * np.exp(-2 * (en * ks_statistic) ** 2)
            p_value = min(1.0, float(p_value))
        except:
            p_value = 0.0

        return ks_statistic, p_value

    def _compute_overlap(
        self, baseline: NDArray[np.float32], comparison: NDArray[np.float32]
    ) -> float:
        """
        Compute distribution overlap coefficient.

        Returns 0-1 where 1 = perfect overlap, 0 = no overlap.

        Uses simplified histogram-based approach.
        """
        # Project to 1D (first principal component approximation)
        baseline_flat = baseline.flatten()[:1000]  # Sample for speed
        comparison_flat = comparison.flatten()[:1000]

        # Create histograms
        all_data = np.concatenate([baseline_flat, comparison_flat])
        bins = np.linspace(all_data.min(), all_data.max(), 50)

        hist1, _ = np.histogram(baseline_flat, bins=bins, density=True)
        hist2, _ = np.histogram(comparison_flat, bins=bins, density=True)

        # Overlap: sum of minimum values
        overlap = float(np.sum(np.minimum(hist1, hist2)) / np.sum(hist1))

        return overlap

    def _classify_severity(
        self, mean_shift: float, ks_statistic: float, overlap: float
    ) -> str:
        """Classify drift severity based on statistics."""
        # High drift: large mean shift OR high KS stat OR low overlap
        if mean_shift > 1.0 or ks_statistic > 0.3 or overlap < 0.5:
            return "high"
        # Medium drift
        elif mean_shift > 0.5 or ks_statistic > 0.15 or overlap < 0.7:
            return "medium"
        # Low drift
        elif mean_shift > 0.2 or ks_statistic > 0.05 or overlap < 0.85:
            return "low"
        else:
            return "none"

    def _classify_drift(self, statistics: DriftStatistics) -> Tuple[bool, float]:
        """
        Determine if drift occurred and confidence level.

        Returns: (drift_detected, confidence)
        """
        # Statistical significance
        statistically_significant = statistics.ks_pvalue < self.significance_level

        # Practical significance
        practically_significant = statistics.drift_severity in ["medium", "high"]

        # Drift detected if both statistical and practical significance
        drift_detected = statistically_significant and practically_significant

        # Confidence based on multiple indicators
        indicators = []

        # KS test confidence
        if statistics.ks_pvalue < 0.01:
            indicators.append(0.95)
        elif statistics.ks_pvalue < 0.05:
            indicators.append(0.75)
        else:
            indicators.append(0.3)

        # Mean shift confidence
        if statistics.mean_shift > 1.0:
            indicators.append(0.9)
        elif statistics.mean_shift > 0.5:
            indicators.append(0.7)
        else:
            indicators.append(0.4)

        # Overlap confidence
        if statistics.overlap_coefficient < 0.5:
            indicators.append(0.9)
        elif statistics.overlap_coefficient < 0.7:
            indicators.append(0.7)
        else:
            indicators.append(0.4)

        confidence = float(np.mean(indicators))

        return drift_detected, confidence

    def _generate_recommendations(
        self, statistics: DriftStatistics, drift_detected: bool
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if not drift_detected:
            recommendations.append("No significant drift detected - embeddings remain stable")
            return recommendations

        # High severity recommendations
        if statistics.drift_severity == "high":
            recommendations.append("CRITICAL: High drift detected - investigate immediately")
            recommendations.append("Check if embedding model version changed")
            recommendations.append("Verify data pipeline for errors or changes")
            recommendations.append("Consider retraining or updating the model")

        # Medium severity recommendations
        elif statistics.drift_severity == "medium":
            recommendations.append("WARNING: Moderate drift detected")
            recommendations.append("Monitor closely for further degradation")
            recommendations.append("Review recent data for topic shifts")

        # Low severity recommendations
        elif statistics.drift_severity == "low":
            recommendations.append("INFO: Minor drift detected")
            recommendations.append("Continue monitoring, no immediate action needed")

        # Specific recommendations based on statistics
        if statistics.std_change > 1.5:
            recommendations.append(
                f"Variance increased {statistics.std_change:.1f}x - embeddings more dispersed"
            )
        elif statistics.std_change < 0.7:
            recommendations.append(
                f"Variance decreased {statistics.std_change:.1f}x - embeddings more concentrated"
            )

        if statistics.overlap_coefficient < 0.6:
            recommendations.append("Low distribution overlap - significant shift in embedding space")

        return recommendations


# ============================================================================
# Singleton
# ============================================================================

_detector_instance: Optional[VectorDriftDetector] = None


def get_detector() -> VectorDriftDetector:
    """Get singleton detector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = VectorDriftDetector()
    return _detector_instance
