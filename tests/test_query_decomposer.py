"""
Tests for Query Decomposer
"""
import asyncio
import pytest
from app.services.query_decomposer import QueryDecomposer, QueryChunk


@pytest.fixture
def decomposer():
    return QueryDecomposer(max_chunks=10)


def run_async(coro):
    """Helper to run async tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestQueryDecomposer:

    def test_empty_query(self, decomposer):
        result = run_async(decomposer.decompose(""))
        assert result.total_chunks == 0
        assert result.chunks == []

    def test_simple_query(self, decomposer):
        result = run_async(decomposer.decompose("user authentication"))
        assert result.total_chunks > 0
        texts = [c.text for c in result.chunks]
        # Should extract "user authentication" as a bigram or individual words
        assert any("user" in t or "authentication" in t for t in texts)

    def test_quoted_entity(self, decomposer):
        result = run_async(decomposer.decompose('find the "login_handler" function'))
        assert result.total_chunks > 0
        # Quoted entities should have highest weight
        quoted = [c for c in result.chunks if "login_handler" in c.text]
        assert len(quoted) > 0
        assert quoted[0].weight >= 0.9  # Quoted = highest priority
        assert quoted[0].intent == "entity_lookup"

    def test_technical_identifiers(self, decomposer):
        result = run_async(decomposer.decompose("check app.services.auth_module"))
        assert result.total_chunks > 0
        # Dotted path should be extracted as entity
        identifiers = [c for c in result.chunks if "app.services.auth_module" in c.text]
        assert len(identifiers) > 0
        assert identifiers[0].intent == "entity_lookup"

    def test_uppercase_constants(self, decomposer):
        result = run_async(decomposer.decompose("what is MAX_RETRY_COUNT"))
        assert result.total_chunks > 0
        constants = [c for c in result.chunks if "MAX_RETRY_COUNT" in c.text]
        assert len(constants) > 0

    def test_stop_words_removed(self, decomposer):
        result = run_async(decomposer.decompose("what is the status of all users"))
        texts = [c.text.lower() for c in result.chunks]
        # Stop words like "what", "is", "the", "of", "all" should be removed
        for t in texts:
            words = t.split()
            assert "what" not in words
            assert "the" not in words

    def test_relationship_intent(self, decomposer):
        result = run_async(decomposer.decompose("relationship between users and orders"))
        rel_chunks = [c for c in result.chunks if c.intent == "relationship_query"]
        assert len(rel_chunks) > 0

    def test_max_chunks_limit(self, decomposer):
        long_query = " ".join([f"word{i}" for i in range(50)])
        result = run_async(decomposer.decompose(long_query))
        assert result.total_chunks <= 10

    def test_weight_ordering(self, decomposer):
        result = run_async(decomposer.decompose("find auth_service.login method details"))
        if len(result.chunks) > 1:
            # Chunks should be sorted by weight descending
            weights = [c.weight for c in result.chunks]
            assert weights == sorted(weights, reverse=True)

    def test_deduplication(self, decomposer):
        result = run_async(decomposer.decompose("user user user authentication"))
        texts = [c.text.lower() for c in result.chunks]
        # Should not have duplicate "user" entries
        unique_texts = set(texts)
        assert len(unique_texts) == len(texts)

    def test_decomposition_time(self, decomposer):
        result = run_async(decomposer.decompose("complex query with many words about authentication"))
        assert result.decomposition_time_ms >= 0
        assert result.decomposition_time_ms < 100  # Should be fast (< 100ms)
