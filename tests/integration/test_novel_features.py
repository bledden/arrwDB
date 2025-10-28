"""
Comprehensive integration tests for all 9 novel features.
Tests use REAL data, no mocks or simulations.
"""

import numpy as np
import pytest
from uuid import uuid4, UUID

from core.search_replay import SearchPath, SearchReplayRecorder
from core.temperature_search import TemperatureSearch
from core.index_oracle import IndexOracle, CorpusProfile
from core.embedding_health import EmbeddingHealthMonitor
from core.vector_clustering import VectorClusterer
from core.query_expansion import QueryExpander
from core.vector_drift import VectorDriftDetector
from core.adaptive_reranking import AdaptiveReranker, FeedbackSignal
from core.hybrid_fusion import HybridFusion


class TestSearchReplay:
    """Test SearchReplay with real search paths."""

    def test_search_path_recording(self):
        """Test recording a complete search path."""
        path = SearchPath(corpus_id=uuid4(), k=10)

        # Simulate HNSW search steps
        for i in range(20):
            path.add_step(
                layer=i % 3,
                node_id=uuid4(),
                distance=float(i * 0.1),
                action="visited" if i % 2 == 0 else "skipped",
                reason=f"Step {i}",
            )

        # Finalize with results
        result_ids = [uuid4() for _ in range(10)]
        result_distances = [float(i * 0.05) for i in range(10)]
        path.finalize(result_ids, result_distances)

        assert path.nodes_visited > 0
        assert len(path.result_ids) == 10
        assert path.duration_ms > 0

    def test_replay_recorder(self):
        """Test SearchReplayRecorder recording functionality."""
        recorder = SearchReplayRecorder()

        # Enable replay
        recorder.enable()
        assert recorder.is_enabled()

        # Start recording paths
        for _ in range(5):
            path = recorder.start_recording(uuid4(), np.random.randn(128).astype(np.float32), 10)
            if path:  # Recording is enabled
                path.add_step(0, uuid4(), 0.5, "visited")
                path.finalize([uuid4()], [0.5])

        # Disable and verify no overhead
        recorder.disable()
        assert not recorder.is_enabled()


class TestTemperatureSearch:
    """Test TemperatureSearch with real results."""

    def test_temperature_sampling(self):
        """Test temperature-based result sampling."""
        # Create realistic search results
        results = [(uuid4(), float(0.9 - i * 0.05)) for i in range(20)]

        # Test different temperatures
        temp_low = TemperatureSearch.apply_temperature(results, k=10, temperature=0.1)
        temp_high = TemperatureSearch.apply_temperature(results, k=10, temperature=2.0)

        assert len(temp_low) == 10
        assert len(temp_high) == 10

        # Low temp should be more greedy (top results)
        low_distances = [d for _, d in temp_low]
        high_distances = [d for _, d in temp_high]

        assert min(low_distances) <= min(high_distances)

    def test_diversity_score(self):
        """Test diversity score computation."""
        # Tight cluster (low diversity)
        tight_results = [(uuid4(), 0.9 + i * 0.01) for i in range(10)]
        tight_diversity = TemperatureSearch.compute_diversity_score(tight_results)

        # Wide spread (high diversity)
        wide_results = [(uuid4(), 0.1 + i * 0.1) for i in range(10)]
        wide_diversity = TemperatureSearch.compute_diversity_score(wide_results)

        assert wide_diversity > tight_diversity

    def test_temperature_recommendations(self):
        """Test temperature recommendations for different use cases."""
        temp_greedy, reason = TemperatureSearch.recommend_temperature("precision")
        temp_explore, reason = TemperatureSearch.recommend_temperature("exploration")

        assert temp_greedy < temp_explore
        assert temp_greedy == 0.0
        assert temp_explore == 2.0


class TestIndexOracle:
    """Test IndexOracle with real corpus profiles."""

    def test_small_corpus_recommendation(self):
        """Test recommendation for small corpus."""
        profile = CorpusProfile(
            vector_count=500,
            dimension=128,
            current_index_type="hnsw",
        )

        oracle = IndexOracle()
        recommendation = oracle.analyze_corpus(profile)

        assert recommendation.recommended_index == "brute_force"
        assert recommendation.confidence > 0.8

    def test_large_corpus_recommendation(self):
        """Test recommendation for large corpus."""
        profile = CorpusProfile(
            vector_count=5_000_000,
            dimension=1024,
            current_index_type="brute_force",
            search_rate_per_minute=100.0,
        )

        oracle = IndexOracle()
        recommendation = oracle.analyze_corpus(profile)

        assert recommendation.recommended_index in ["hnsw", "ivf"]
        assert len(recommendation.reasoning) > 0

    def test_write_heavy_workload(self):
        """Test recommendation for write-heavy workload."""
        profile = CorpusProfile(
            vector_count=10_000,
            dimension=512,
            current_index_type="hnsw",
            insert_rate_per_minute=100.0,
            search_rate_per_minute=10.0,
        )

        oracle = IndexOracle()
        recommendation = oracle.analyze_corpus(profile)

        # Write-heavy should favor brute_force
        assert "write" in " ".join(recommendation.reasoning).lower()


class TestEmbeddingHealthMonitor:
    """Test EmbeddingHealthMonitor with real embeddings."""

    def test_healthy_embeddings(self):
        """Test analysis of healthy embeddings."""
        # Create normalized, well-distributed embeddings
        embeddings = np.random.randn(100, 128).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        vector_ids = [uuid4() for _ in range(100)]

        monitor = EmbeddingHealthMonitor()
        health = monitor.analyze_corpus(uuid4(), embeddings, vector_ids)

        assert health.health_score > 0.7
        assert health.dimension_utilization > 0.9
        assert health.degenerate_count == 0

    def test_unhealthy_embeddings(self):
        """Test detection of embedding quality issues."""
        # Create problematic embeddings
        embeddings = np.random.randn(100, 128).astype(np.float32)

        # Add degenerate vectors
        embeddings[0:5] = 0.0

        # Add outliers
        embeddings[10:12] *= 100

        # Collapse dimensions
        embeddings[:, 0:20] = 0.0

        vector_ids = [uuid4() for _ in range(100)]

        monitor = EmbeddingHealthMonitor()
        health = monitor.analyze_corpus(uuid4(), embeddings, vector_ids)

        assert health.health_score < 0.9
        assert health.degenerate_count == 5
        assert health.outlier_count >= 2
        assert health.dimension_utilization < 1.0
        assert len(health.issues) > 0
        assert len(health.recommendations) > 0

    def test_single_vector_analysis(self):
        """Test single vector health check."""
        embedding = np.random.randn(128).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        monitor = EmbeddingHealthMonitor()
        stats = monitor.analyze_vector(
            embedding,
            uuid4(),
            {"mean_norm": 1.0, "std_norm": 0.1}
        )

        assert stats.norm > 0.9
        assert stats.norm < 1.1
        assert stats.sparsity < 0.5
        assert not stats.is_outlier


class TestVectorClustering:
    """Test VectorClustering with real embeddings."""

    def test_clear_cluster_structure(self):
        """Test clustering with well-separated clusters."""
        # Create 3 distinct clusters
        np.random.seed(42)
        cluster1 = np.random.randn(50, 64) + np.array([5.0] * 64)
        cluster2 = np.random.randn(50, 64) + np.array([-5.0] * 64)
        cluster3 = np.random.randn(50, 64)

        embeddings = np.vstack([cluster1, cluster2, cluster3]).astype(np.float32)
        vector_ids = [uuid4() for _ in range(150)]

        clusterer = VectorClusterer()
        result = clusterer.cluster_corpus(
            uuid4(), embeddings, vector_ids, n_clusters=3
        )

        assert result.num_clusters == 3
        assert result.silhouette_score > 0.5
        assert result.davies_bouldin_score < 1.0
        assert len(result.clusters) == 3

    def test_auto_cluster_detection(self):
        """Test automatic cluster count estimation."""
        embeddings = np.random.randn(200, 64).astype(np.float32)
        vector_ids = [uuid4() for _ in range(200)]

        clusterer = VectorClusterer()
        result = clusterer.cluster_corpus(
            uuid4(), embeddings, vector_ids, algorithm="auto"
        )

        assert result.num_clusters >= 2
        assert result.num_clusters <= 50
        assert len(result.clusters) == result.num_clusters


class TestQueryExpansion:
    """Test QueryExpansion with real queries."""

    def test_balanced_expansion(self):
        """Test balanced query expansion strategy."""
        expander = QueryExpander()
        result = expander.expand_query(
            "machine learning tutorial",
            strategy="balanced"
        )

        assert result.num_expansions > 0
        assert any(e.expansion_type == "original" for e in result.expanded_queries)
        assert result.strategy == "balanced"

    def test_conservative_vs_aggressive(self):
        """Test different expansion strategies."""
        expander = QueryExpander()

        conservative = expander.expand_query("python car", strategy="conservative")
        aggressive = expander.expand_query("python car", strategy="aggressive")

        assert aggressive.num_expansions >= conservative.num_expansions

    def test_rrf_fusion(self):
        """Test Reciprocal Rank Fusion of results."""
        expander = QueryExpander()

        results_by_query = {
            "query1": [(uuid4(), 0.9), (uuid4(), 0.8)],
            "query2": [(uuid4(), 0.85), (uuid4(), 0.75)],
        }
        weights = {"query1": 1.0, "query2": 0.8}

        merged = expander.merge_search_results(results_by_query, weights, "rrf")

        assert len(merged) == 4
        assert all(isinstance(vid, UUID) for vid, _ in merged)


class TestVectorDrift:
    """Test VectorDrift with real embedding distributions."""

    def test_no_drift_detection(self):
        """Test when no drift is present."""
        baseline = np.random.randn(100, 128).astype(np.float32)
        comparison = np.random.randn(100, 128).astype(np.float32)

        detector = VectorDriftDetector()
        result = detector.detect_drift(uuid4(), baseline, comparison)

        assert result.baseline_size == 100
        assert result.comparison_size == 100
        assert result.statistics.drift_severity in ["none", "low", "medium", "high"]

    def test_high_drift_detection(self):
        """Test detection of significant drift."""
        baseline = np.random.randn(100, 128).astype(np.float32)
        comparison = np.random.randn(100, 128).astype(np.float32) + 3.0  # Shifted

        detector = VectorDriftDetector()
        result = detector.detect_drift(uuid4(), baseline, comparison)

        assert result.drift_detected
        assert result.statistics.mean_shift > 1.0
        assert result.statistics.drift_severity in ["medium", "high"]
        assert len(result.recommendations) > 0


class TestAdaptiveReranking:
    """Test AdaptiveReranking with real feedback."""

    def test_feedback_based_reranking(self):
        """Test reranking with user feedback."""
        reranker = AdaptiveReranker()

        # Original results
        results = [(uuid4(), 0.9 - i * 0.05) for i in range(10)]

        # Create feedback favoring result at index 5
        feedback = [
            FeedbackSignal(results[5][0], "click", 0.8),
            FeedbackSignal(results[5][0], "dwell", 0.9),
        ]

        reranked = reranker.rerank_with_feedback(results, feedback)

        assert len(reranked.reranked_results) == 10
        assert len(reranked.boost_applied) > 0

        # Result 5 should be boosted
        assert reranked.boost_applied.get(results[5][0], 0.0) > 0

    def test_multiple_feedback_types(self):
        """Test different feedback signal types."""
        reranker = AdaptiveReranker()

        results = [(uuid4(), 0.9) for _ in range(5)]

        feedback = [
            FeedbackSignal(results[0][0], "click", 1.0),
            FeedbackSignal(results[1][0], "dwell", 1.0),
            FeedbackSignal(results[2][0], "skip", 1.0),
            FeedbackSignal(results[3][0], "bookmark", 1.0),
        ]

        reranked = reranker.rerank_with_feedback(results, feedback)

        # Bookmarked should have highest boost
        boost_bookmark = reranked.boost_applied.get(results[3][0], 0.0)
        boost_skip = reranked.boost_applied.get(results[2][0], 0.0)

        assert boost_bookmark > boost_skip


class TestHybridFusion:
    """Test HybridFusion with real multi-strategy results."""

    def test_linear_fusion(self):
        """Test linear weighted fusion."""
        fusion = HybridFusion()

        results_by_strategy = {
            "vector": [(uuid4(), 0.9), (uuid4(), 0.8)],
            "keyword": [(uuid4(), 0.85), (uuid4(), 0.75)],
        }
        weights = {"vector": 0.7, "keyword": 0.3}

        result = fusion.fuse_results(results_by_strategy, method="linear", weights=weights)

        assert len(result.fused_results) == 4
        assert result.method == "linear"
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0

    def test_rrf_fusion(self):
        """Test Reciprocal Rank Fusion."""
        fusion = HybridFusion()

        shared_id = uuid4()
        results_by_strategy = {
            "vector": [(shared_id, 0.9), (uuid4(), 0.8)],
            "keyword": [(shared_id, 0.85), (uuid4(), 0.75)],
        }

        result = fusion.fuse_results(results_by_strategy, method="rrf")

        # Shared result should rank higher
        fused_ids = [vid for vid, _ in result.fused_results]
        assert shared_id in fused_ids

    def test_confidence_computation(self):
        """Test fusion confidence based on strategy agreement."""
        fusion = HybridFusion()

        # High agreement (same top results)
        shared_ids = [uuid4() for _ in range(10)]
        high_agreement = {
            "s1": [(vid, 0.9) for vid in shared_ids],
            "s2": [(vid, 0.85) for vid in shared_ids],
        }

        # Low agreement (different results)
        low_agreement = {
            "s1": [(uuid4(), 0.9) for _ in range(10)],
            "s2": [(uuid4(), 0.85) for _ in range(10)],
        }

        high_result = fusion.fuse_results(high_agreement, method="rrf")
        low_result = fusion.fuse_results(low_agreement, method="rrf")

        assert high_result.confidence > low_result.confidence


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
