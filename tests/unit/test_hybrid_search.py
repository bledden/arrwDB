"""
Test suite for app/services/hybrid_search.py

Coverage targets:
- ScoringConfig validation
- HybridSearchScorer with various weight combinations
- Metadata scoring (field boosts, recency)
- ResultReranker with custom functions
- Pre-built reranking functions (recency, position, length)
- Score normalization and breakdown
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from app.services.hybrid_search import (
    ScoringConfig,
    HybridSearchScorer,
    ResultReranker,
    boost_by_recency,
    boost_by_chunk_position,
    boost_by_length,
)
from app.models.base import Chunk, ChunkMetadata


class TestScoringConfig:
    """Test ScoringConfig validation and initialization."""

    def test_scoring_config_default(self):
        """Test default scoring configuration."""
        config = ScoringConfig()
        assert config.vector_weight == 0.7
        assert config.metadata_weight == 0.3
        assert config.field_boosts is None
        assert config.recency_boost_enabled is False
        assert config.normalize_scores is True

    def test_scoring_config_custom_weights(self):
        """Test custom weight configuration."""
        config = ScoringConfig(vector_weight=0.5, metadata_weight=0.5)
        assert config.vector_weight == 0.5
        assert config.metadata_weight == 0.5

    def test_scoring_config_validation_vector_weight_too_low(self):
        """Test validation fails when vector_weight < 0."""
        with pytest.raises(ValueError, match="vector_weight must be between 0 and 1"):
            ScoringConfig(vector_weight=-0.1, metadata_weight=1.1)

    def test_scoring_config_validation_vector_weight_too_high(self):
        """Test validation fails when vector_weight > 1."""
        with pytest.raises(ValueError, match="vector_weight must be between 0 and 1"):
            ScoringConfig(vector_weight=1.5, metadata_weight=-0.5)

    def test_scoring_config_validation_metadata_weight_too_low(self):
        """Test validation fails when metadata_weight < 0."""
        with pytest.raises(ValueError, match="metadata_weight must be between 0 and 1"):
            ScoringConfig(vector_weight=0.5, metadata_weight=-0.1)

    def test_scoring_config_validation_metadata_weight_too_high(self):
        """Test validation fails when metadata_weight > 1."""
        with pytest.raises(ValueError, match="metadata_weight must be between 0 and 1"):
            ScoringConfig(vector_weight=0.5, metadata_weight=1.5)

    def test_scoring_config_validation_weights_dont_sum_to_one(self):
        """Test validation fails when weights don't sum to 1.0."""
        with pytest.raises(ValueError, match="vector_weight \\+ metadata_weight must equal 1.0"):
            ScoringConfig(vector_weight=0.6, metadata_weight=0.5)

    def test_scoring_config_with_field_boosts(self):
        """Test configuration with field boosts."""
        config = ScoringConfig(
            vector_weight=0.7,
            metadata_weight=0.3,
            field_boosts={"tags": 2.0, "author": 1.5}
        )
        assert config.field_boosts == {"tags": 2.0, "author": 1.5}

    def test_scoring_config_with_recency(self):
        """Test configuration with recency boost."""
        config = ScoringConfig(
            recency_boost_enabled=True,
            recency_half_life_days=60.0
        )
        assert config.recency_boost_enabled is True
        assert config.recency_half_life_days == 60.0


class TestHybridSearchScorer:
    """Test HybridSearchScorer functionality."""

    @pytest.fixture
    def mock_chunks(self):
        """Create mock chunks with various ages and positions."""
        chunks = []

        # Recent chunk (created today)
        chunks.append(Chunk(
            id=uuid4(),
            text="Recent chunk with test content",
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
            metadata=ChunkMetadata(
                source_document_id=uuid4(),
                chunk_index=0,
                created_at=datetime.utcnow()
            )
        ))

        # Old chunk (created 60 days ago)
        chunks.append(Chunk(
            id=uuid4(),
            text="Old chunk",
            embedding=[0.2, 0.3, 0.4, 0.5, 0.6],
            metadata=ChunkMetadata(
                source_document_id=uuid4(),
                chunk_index=5,
                created_at=datetime.utcnow() - timedelta(days=60)
            )
        ))

        # Middle-aged chunk (created 30 days ago)
        chunks.append(Chunk(
            id=uuid4(),
            text="Middle-aged chunk with moderate age",
            embedding=[0.3, 0.4, 0.5, 0.6, 0.7],
            metadata=ChunkMetadata(
                source_document_id=uuid4(),
                chunk_index=2,
                created_at=datetime.utcnow() - timedelta(days=30)
            )
        ))

        return chunks

    def test_hybrid_scorer_initialization_default(self):
        """Test scorer initialization with default config."""
        scorer = HybridSearchScorer()
        assert scorer.config.vector_weight == 0.7
        assert scorer.config.metadata_weight == 0.3

    def test_hybrid_scorer_initialization_custom(self):
        """Test scorer initialization with custom config."""
        config = ScoringConfig(vector_weight=0.6, metadata_weight=0.4)
        scorer = HybridSearchScorer(config)
        assert scorer.config.vector_weight == 0.6
        assert scorer.config.metadata_weight == 0.4

    def test_score_results_empty_list(self):
        """Test scoring with empty results list."""
        scorer = HybridSearchScorer()
        results = scorer.score_results([])
        assert results == []

    def test_score_results_basic(self, mock_chunks):
        """Test basic scoring without metadata boosts."""
        scorer = HybridSearchScorer()

        # Create search results: (chunk, distance)
        search_results = [
            (mock_chunks[0], 0.2),  # Close match (low distance)
            (mock_chunks[1], 0.8),  # Far match (high distance)
            (mock_chunks[2], 0.5),  # Medium match
        ]

        results = scorer.score_results(search_results)

        # Should return 3 results
        assert len(results) == 3

        # Each result should have (chunk, score, breakdown)
        for chunk, score, breakdown in results:
            assert isinstance(chunk, Chunk)
            assert 0.0 <= score <= 1.0
            assert "vector_score" in breakdown
            assert "metadata_score" in breakdown
            assert "hybrid_score" in breakdown

        # Results should be sorted by score (descending)
        scores = [score for _, score, _ in results]
        assert scores == sorted(scores, reverse=True)

    def test_score_results_vector_similarity_conversion(self, mock_chunks):
        """Test that distance is correctly converted to similarity."""
        scorer = HybridSearchScorer()

        # Distance 0.0 should give vector_score close to 1.0
        results = scorer.score_results([(mock_chunks[0], 0.0)])
        _, score, breakdown = results[0]
        assert breakdown["vector_score"] == 1.0

        # Distance 2.0 should give vector_score close to 0.0
        results = scorer.score_results([(mock_chunks[0], 2.0)])
        _, score, breakdown = results[0]
        assert breakdown["vector_score"] == 0.0

    def test_score_results_with_recency_boost(self, mock_chunks):
        """Test scoring with recency boost enabled."""
        config = ScoringConfig(
            vector_weight=0.7,
            metadata_weight=0.3,
            recency_boost_enabled=True,
            recency_half_life_days=30.0
        )
        scorer = HybridSearchScorer(config)

        # Recent chunk vs old chunk (same vector distance)
        search_results = [
            (mock_chunks[0], 0.5),  # Recent (created today)
            (mock_chunks[1], 0.5),  # Old (created 60 days ago)
        ]

        results = scorer.score_results(search_results)

        # Recent chunk should score higher
        recent_score = results[0][1]
        old_score = results[1][1]
        assert recent_score > old_score

        # Check breakdown includes recency_boost
        assert "recency_boost" in results[0][2]

    def test_score_results_weight_distribution(self, mock_chunks):
        """Test that weights are correctly applied."""
        # 100% vector weight
        config = ScoringConfig(vector_weight=1.0, metadata_weight=0.0)
        scorer = HybridSearchScorer(config)

        results = scorer.score_results([(mock_chunks[0], 0.4)])
        _, score, breakdown = results[0]

        # Hybrid score should equal vector score when metadata_weight = 0
        assert abs(score - breakdown["vector_score"]) < 0.01

    def test_score_results_breakdown_completeness(self, mock_chunks):
        """Test that score breakdown contains all expected fields."""
        scorer = HybridSearchScorer()

        results = scorer.score_results([(mock_chunks[0], 0.3)])
        _, score, breakdown = results[0]

        # Verify all expected fields
        assert "vector_score" in breakdown
        assert "vector_distance" in breakdown
        assert "metadata_score" in breakdown
        assert "hybrid_score" in breakdown
        assert "vector_weight" in breakdown
        assert "metadata_weight" in breakdown

        assert breakdown["vector_distance"] == 0.3
        assert breakdown["vector_weight"] == 0.7
        assert breakdown["metadata_weight"] == 0.3


class TestRecencyBoost:
    """Test recency boost calculation."""

    def test_recency_boost_fresh_document(self):
        """Test recency boost for document created today."""
        config = ScoringConfig(
            recency_boost_enabled=True,
            recency_half_life_days=30.0
        )
        scorer = HybridSearchScorer(config)

        # Create fresh chunk
        chunk = Chunk(
            id=uuid4(),
            text="Fresh content",
            embedding=[0.1, 0.2, 0.3],
            metadata=ChunkMetadata(
                source_document_id=uuid4(),
                chunk_index=0,
                created_at=datetime.utcnow()
            )
        )

        score = scorer._calculate_recency_boost(chunk)

        # Fresh document should have score close to 1.0
        assert score >= 0.99

    def test_recency_boost_half_life(self):
        """Test recency boost at half-life age."""
        config = ScoringConfig(
            recency_boost_enabled=True,
            recency_half_life_days=30.0
        )
        scorer = HybridSearchScorer(config)

        # Create chunk at half-life age
        chunk = Chunk(
            id=uuid4(),
            text="Half-life content",
            embedding=[0.1, 0.2, 0.3],
            metadata=ChunkMetadata(
                source_document_id=uuid4(),
                chunk_index=0,
                created_at=datetime.utcnow() - timedelta(days=30)
            )
        )

        score = scorer._calculate_recency_boost(chunk)

        # At half-life, score should be ~0.5
        assert 0.45 <= score <= 0.55

    def test_recency_boost_old_document(self):
        """Test recency boost for very old document."""
        config = ScoringConfig(
            recency_boost_enabled=True,
            recency_half_life_days=30.0
        )
        scorer = HybridSearchScorer(config)

        # Create very old chunk (120 days = 4 half-lives)
        chunk = Chunk(
            id=uuid4(),
            text="Old content",
            embedding=[0.1, 0.2, 0.3],
            metadata=ChunkMetadata(
                source_document_id=uuid4(),
                chunk_index=0,
                created_at=datetime.utcnow() - timedelta(days=120)
            )
        )

        score = scorer._calculate_recency_boost(chunk)

        # At 4 half-lives, score should be ~0.0625 (1/16)
        assert 0.05 <= score <= 0.08


class TestResultReranker:
    """Test ResultReranker functionality."""

    @pytest.fixture
    def mock_chunks(self):
        """Create mock chunks for reranking."""
        return [
            Chunk(
                id=uuid4(),
                text="Short",
                embedding=[0.1, 0.2, 0.3],
                metadata=ChunkMetadata(
                    source_document_id=uuid4(),
                    chunk_index=0,
                    created_at=datetime.utcnow()
                )
            ),
            Chunk(
                id=uuid4(),
                text="Medium length text with some content",
                embedding=[0.2, 0.3, 0.4],
                metadata=ChunkMetadata(
                    source_document_id=uuid4(),
                    chunk_index=10,
                    created_at=datetime.utcnow() - timedelta(days=60)
                )
            ),
            Chunk(
                id=uuid4(),
                text="Very long text " * 50,  # Long text
                embedding=[0.3, 0.4, 0.5],
                metadata=ChunkMetadata(
                    source_document_id=uuid4(),
                    chunk_index=2,
                    created_at=datetime.utcnow() - timedelta(days=30)
                )
            ),
        ]

    def test_reranker_initialization(self):
        """Test reranker initialization with custom function."""
        def custom_fn(chunk, score):
            return score * 2.0

        reranker = ResultReranker(custom_fn)
        assert reranker.scoring_fn == custom_fn

    def test_reranker_basic(self, mock_chunks):
        """Test basic reranking with custom function."""
        # Simple function that doubles the score
        def double_score(chunk, score):
            return score * 2.0

        reranker = ResultReranker(double_score)

        results = [
            (mock_chunks[0], 0.5),
            (mock_chunks[1], 0.3),
            (mock_chunks[2], 0.7),
        ]

        reranked = reranker.rerank(results)

        # Should return same number of results
        assert len(reranked) == 3

        # Scores should be doubled
        assert reranked[0][1] == 1.4  # 0.7 * 2
        assert reranked[1][1] == 1.0  # 0.5 * 2
        assert reranked[2][1] == 0.6  # 0.3 * 2

    def test_reranker_changes_order(self, mock_chunks):
        """Test that reranking can change result order."""
        # Function that boosts shorter text
        def boost_short(chunk, score):
            length_penalty = len(chunk.text) / 1000.0
            return score + (1.0 - length_penalty)

        reranker = ResultReranker(boost_short)

        results = [
            (mock_chunks[2], 0.8),  # Long text, high score
            (mock_chunks[0], 0.5),  # Short text, medium score
        ]

        reranked = reranker.rerank(results)

        # Order may change based on length boost
        # Short text should get significant boost
        assert reranked[0][0].text == "Short"


class TestPrebuiltRerankers:
    """Test pre-built reranking functions."""

    @pytest.fixture
    def mock_chunks(self):
        """Create mock chunks with various characteristics."""
        return [
            Chunk(
                id=uuid4(),
                text="Short recent early",
                embedding=[0.1, 0.2, 0.3],
                metadata=ChunkMetadata(
                    source_document_id=uuid4(),
                    chunk_index=0,
                    created_at=datetime.utcnow()
                )
            ),
            Chunk(
                id=uuid4(),
                text="Long " * 500,  # Very long text
                embedding=[0.2, 0.3, 0.4],
                metadata=ChunkMetadata(
                    source_document_id=uuid4(),
                    chunk_index=50,
                    created_at=datetime.utcnow() - timedelta(days=60)
                )
            ),
            Chunk(
                id=uuid4(),
                text="Medium length content here",
                embedding=[0.3, 0.4, 0.5],
                metadata=ChunkMetadata(
                    source_document_id=uuid4(),
                    chunk_index=10,
                    created_at=datetime.utcnow() - timedelta(days=30)
                )
            ),
        ]

    def test_boost_by_recency_function(self, mock_chunks):
        """Test boost_by_recency reranking function."""
        rerank_fn = boost_by_recency(half_life_days=30.0)

        # Recent chunk should get higher score
        recent_score = rerank_fn(mock_chunks[0], 0.5)
        old_score = rerank_fn(mock_chunks[1], 0.5)

        assert recent_score > old_score

    def test_boost_by_chunk_position_prefer_early(self, mock_chunks):
        """Test boost_by_chunk_position with prefer_early=True."""
        rerank_fn = boost_by_chunk_position(prefer_early=True)

        # Earlier chunk should get higher score
        early_score = rerank_fn(mock_chunks[0], 0.5)  # chunk_index=0
        late_score = rerank_fn(mock_chunks[1], 0.5)   # chunk_index=50

        assert early_score > late_score

    def test_boost_by_chunk_position_prefer_late(self, mock_chunks):
        """Test boost_by_chunk_position with prefer_early=False."""
        rerank_fn = boost_by_chunk_position(prefer_early=False)

        # Later chunk should get higher score
        early_score = rerank_fn(mock_chunks[0], 0.5)  # chunk_index=0
        late_score = rerank_fn(mock_chunks[1], 0.5)   # chunk_index=50

        assert late_score > early_score

    def test_boost_by_length_prefer_longer(self, mock_chunks):
        """Test boost_by_length with prefer_longer=True."""
        rerank_fn = boost_by_length(prefer_longer=True)

        # Longer chunk should get higher score
        short_score = rerank_fn(mock_chunks[0], 0.5)  # Short text
        long_score = rerank_fn(mock_chunks[1], 0.5)   # Long text

        assert long_score > short_score

    def test_boost_by_length_prefer_shorter(self, mock_chunks):
        """Test boost_by_length with prefer_longer=False."""
        rerank_fn = boost_by_length(prefer_longer=False)

        # Shorter chunk should get higher score
        short_score = rerank_fn(mock_chunks[0], 0.5)  # Short text
        long_score = rerank_fn(mock_chunks[1], 0.5)   # Long text

        assert short_score > long_score

    def test_prebuilt_reranker_with_result_reranker(self, mock_chunks):
        """Test using pre-built functions with ResultReranker."""
        # Use boost_by_recency with ResultReranker
        rerank_fn = boost_by_recency(half_life_days=30.0)
        reranker = ResultReranker(rerank_fn)

        results = [
            (mock_chunks[1], 0.8),  # Old chunk, high initial score
            (mock_chunks[0], 0.7),  # Recent chunk, slightly lower score
        ]

        reranked = reranker.rerank(results)

        # Recent chunk should be boosted above old chunk
        assert reranked[0][0] == mock_chunks[0]  # Recent chunk now first


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_hybrid_scorer_with_invalid_distance(self):
        """Test scorer handles invalid distance values gracefully."""
        scorer = HybridSearchScorer()

        chunk = Chunk(
            id=uuid4(),
            text="Test",
            embedding=[0.1, 0.2, 0.3],
            metadata=ChunkMetadata(
                source_document_id=uuid4(),
                chunk_index=0
            )
        )

        # Negative distance (should clamp to valid range)
        results = scorer.score_results([(chunk, -0.5)])
        _, score, breakdown = results[0]
        assert 0.0 <= breakdown["vector_score"] <= 1.0

        # Distance > 2.0 (should clamp to valid range)
        results = scorer.score_results([(chunk, 5.0)])
        _, score, breakdown = results[0]
        assert 0.0 <= breakdown["vector_score"] <= 1.0

    def test_reranker_with_error_in_scoring_function(self):
        """Test reranker handles errors in custom scoring function."""
        def buggy_fn(chunk, score):
            if len(chunk.text) < 10:
                raise ValueError("Text too short")
            return score

        reranker = ResultReranker(buggy_fn)

        chunk = Chunk(
            id=uuid4(),
            text="Short",
            embedding=[0.1, 0.2],
            metadata=ChunkMetadata(
                source_document_id=uuid4(),
                chunk_index=0
            )
        )

        # Should raise the error (no error handling in reranker)
        with pytest.raises(ValueError, match="Text too short"):
            reranker.rerank([(chunk, 0.5)])

    def test_prebuilt_rerankers_handle_missing_metadata(self):
        """Test pre-built rerankers handle chunks with minimal metadata."""
        chunk = Chunk(
            id=uuid4(),
            text="Test content",
            embedding=[0.1, 0.2, 0.3],
            metadata=ChunkMetadata(
                source_document_id=uuid4(),
                chunk_index=0
            )
        )

        # All pre-built rerankers should handle this gracefully
        recency_fn = boost_by_recency()
        position_fn = boost_by_chunk_position()
        length_fn = boost_by_length()

        # Should not raise errors
        recency_score = recency_fn(chunk, 0.5)
        position_score = position_fn(chunk, 0.5)
        length_score = length_fn(chunk, 0.5)

        # All should return valid scores
        assert 0.0 <= recency_score <= 1.0
        assert 0.0 <= position_score <= 1.0
        assert 0.0 <= length_score <= 1.0
