"""
Data Vent - Result Aggregator

Merges parallel search results into a unified, ranked response.
Handles deduplication, score fusion, cross-chunk boosting, and completion checking.
"""

import structlog
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Set
from collections import Counter

from app.services.intelligent_retriever import ChunkNode
from app.services.parallel_search import ChunkSearchResult, ParallelSearchResult

logger = structlog.get_logger()


# ─── Scoring weights ────────────────────────────────────────────────────────

VECTOR_SCORE_WEIGHT = 0.6
GRAPH_SCORE_WEIGHT = 0.3
CROSS_CHUNK_BOOST_WEIGHT = 0.1


@dataclass
class ScoredChunk:
    """A chunk with aggregated scoring from multiple search results."""

    chunk_id: str
    content: str
    final_score: float
    vector_score: float = 0.0
    graph_score: float = 0.0
    cross_chunk_boost: float = 0.0
    chunk_type: str = ""
    source_id: str = ""
    document_id: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)
    matched_by_chunks: List[str] = field(default_factory=list)
    depth: int = 0


@dataclass
class AggregatedResult:
    """Final aggregated search result."""

    chunks: List[ScoredChunk]
    total_results: int
    unique_sources: int
    vector_matches: int
    graph_matches: int
    completion_reached: bool
    aggregation_time_ms: float = 0.0


class ResultAggregator:
    """
    Merges parallel chunk search results into a unified ranked response.

    Pipeline:
    1. Collect all chunks from all parallel results
    2. Deduplicate by chunk_id
    3. Fuse scores: vector_score × 0.6 + graph_score × 0.3 + cross_chunk_boost × 0.1
    4. Cross-chunk boosting: nodes found by multiple chunks get score boost
    5. Rank by final_score
    6. Check completion
    """

    def __init__(
        self,
        max_results: int = 50,
        min_avg_score: float = 0.5,
        min_chunks_for_completion: int = 3,
        vector_weight: float = VECTOR_SCORE_WEIGHT,
        graph_weight: float = GRAPH_SCORE_WEIGHT,
        cross_chunk_weight: float = CROSS_CHUNK_BOOST_WEIGHT,
    ):
        self.max_results = max_results
        self.min_avg_score = min_avg_score
        self.min_chunks_for_completion = min_chunks_for_completion
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight
        self.cross_chunk_weight = cross_chunk_weight

    async def aggregate(
        self,
        parallel_result: ParallelSearchResult,
        original_query: str = "",
        limit: int = 0,
    ) -> AggregatedResult:
        """
        Aggregate all parallel search results into a single ranked list.
        """
        import time
        start = time.perf_counter()

        limit = limit or self.max_results

        # Step 1: Collect all chunks per chunk_id
        chunk_map: Dict[str, _ChunkAccumulator] = {}

        for chunk_result in parallel_result.chunk_results:
            if chunk_result.error:
                continue

            chunk_text = chunk_result.query_chunk.text
            chunk_weight = chunk_result.query_chunk.weight

            # Vector results
            for node in chunk_result.vector_results:
                acc = chunk_map.setdefault(
                    node.chunk_id, _ChunkAccumulator(node=node)
                )
                acc.vector_scores.append(node.score * chunk_weight)
                acc.matched_by.add(chunk_text)

            # Graph results
            for node in chunk_result.graph_results:
                acc = chunk_map.setdefault(
                    node.chunk_id, _ChunkAccumulator(node=node)
                )
                acc.graph_scores.append(node.score * chunk_weight)
                acc.matched_by.add(chunk_text)

        if not chunk_map:
            elapsed = (time.perf_counter() - start) * 1000
            return AggregatedResult(
                chunks=[],
                total_results=0,
                unique_sources=0,
                vector_matches=parallel_result.total_vector_hits,
                graph_matches=parallel_result.total_graph_hits,
                completion_reached=False,
                aggregation_time_ms=round(elapsed, 2),
            )

        # Step 2: Compute fused scores
        total_query_chunks = max(parallel_result.chunks_searched, 1)
        scored: List[ScoredChunk] = []

        for chunk_id, acc in chunk_map.items():
            # Best vector score from any chunk match
            best_vector = max(acc.vector_scores) if acc.vector_scores else 0.0
            # Best graph score
            best_graph = max(acc.graph_scores) if acc.graph_scores else 0.0
            # Cross-chunk boost: how many different query chunks matched this node
            cross_boost = len(acc.matched_by) / total_query_chunks

            final_score = (
                best_vector * self.vector_weight
                + best_graph * self.graph_weight
                + cross_boost * self.cross_chunk_weight
            )

            scored.append(ScoredChunk(
                chunk_id=chunk_id,
                content=acc.node.content,
                final_score=round(final_score, 4),
                vector_score=round(best_vector, 4),
                graph_score=round(best_graph, 4),
                cross_chunk_boost=round(cross_boost, 4),
                chunk_type=acc.node.chunk_type,
                source_id=acc.node.source_id,
                document_id=acc.node.document_id,
                metadata=acc.node.metadata,
                matched_by_chunks=list(acc.matched_by),
                depth=acc.node.depth,
            ))

        # Step 3: Rank by final score
        scored.sort(key=lambda c: c.final_score, reverse=True)
        scored = scored[:limit]

        # Step 4: Completion check
        completion = self._check_completion(scored, total_query_chunks)

        # Unique sources
        unique_sources = len(set(c.source_id for c in scored if c.source_id))

        elapsed = (time.perf_counter() - start) * 1000

        result = AggregatedResult(
            chunks=scored,
            total_results=len(scored),
            unique_sources=unique_sources,
            vector_matches=parallel_result.total_vector_hits,
            graph_matches=parallel_result.total_graph_hits,
            completion_reached=completion,
            aggregation_time_ms=round(elapsed, 2),
        )

        logger.info(
            "results_aggregated",
            total=len(scored),
            unique_sources=unique_sources,
            completion=completion,
            time_ms=result.aggregation_time_ms,
        )

        return result

    def _check_completion(
        self, chunks: List[ScoredChunk], total_query_chunks: int
    ) -> bool:
        """
        Check if enough high-quality context has been retrieved.

        Conditions for completion:
        - At least min_chunks_for_completion results
        - Average score >= min_avg_score
        - OR result count >= max_results
        """
        if len(chunks) < self.min_chunks_for_completion:
            return False

        if len(chunks) >= self.max_results:
            return True

        avg_score = sum(c.final_score for c in chunks) / max(len(chunks), 1)
        return avg_score >= self.min_avg_score


class _ChunkAccumulator:
    """Internal accumulator for merging scores per chunk_id."""

    __slots__ = ("node", "vector_scores", "graph_scores", "matched_by")

    def __init__(self, node: ChunkNode):
        self.node = node
        self.vector_scores: List[float] = []
        self.graph_scores: List[float] = []
        self.matched_by: Set[str] = set()
