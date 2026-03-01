"""
Data Vent - Parallel Search Dispatcher

Fans out query chunks across FalkorDB in parallel using asyncio.
Each chunk is independently vectorized, searched, and traversed.
"""

import asyncio
import structlog
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from app.services.query_decomposer import QueryChunk
from app.services.intelligent_retriever import IntelligentRetriever, ChunkNode

logger = structlog.get_logger()


@dataclass
class ChunkSearchResult:
    """Results from searching a single query chunk."""

    query_chunk: QueryChunk
    vector_results: List[ChunkNode] = field(default_factory=list)
    graph_results: List[ChunkNode] = field(default_factory=list)
    search_time_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class ParallelSearchResult:
    """Combined results from all parallel chunk searches."""

    chunk_results: List[ChunkSearchResult]
    total_vector_hits: int = 0
    total_graph_hits: int = 0
    total_time_ms: float = 0.0
    chunks_searched: int = 0
    chunks_failed: int = 0


class ParallelSearchDispatcher:
    """
    Dispatches query chunks for parallel search against FalkorDB.

    For each QueryChunk:
    1. Vectorize the chunk text (via embeddings-service)
    2. Vector similarity search against 768-dim nodes
    3. DFS traversal from top matching nodes
    All chunks execute concurrently via asyncio.gather().
    """

    def __init__(
        self,
        per_chunk_timeout: float = 10.0,
        vector_top_k: int = 5,
        dfs_depth: int = 2,
        dfs_min_relevance: float = 0.3,
        dfs_max_results: int = 20,
        batch_vectorize: bool = True,
    ):
        self.per_chunk_timeout = per_chunk_timeout
        self.vector_top_k = vector_top_k
        self.dfs_depth = dfs_depth
        self.dfs_min_relevance = dfs_min_relevance
        self.dfs_max_results = dfs_max_results
        self.batch_vectorize = batch_vectorize

    async def dispatch(
        self,
        chunks: List[QueryChunk],
        retriever: IntelligentRetriever,
    ) -> ParallelSearchResult:
        """
        Run all chunks in parallel. Returns ParallelSearchResult.

        Pipeline:
        1. Batch-vectorize all chunk texts (single HTTP call)
        2. Fan out: for each chunk, run vector search + DFS in parallel
        3. Collect results, handle per-chunk errors gracefully
        """
        start_time = time.perf_counter()

        if not chunks:
            return ParallelSearchResult(
                chunk_results=[],
                total_time_ms=0.0,
            )

        # Step 1: Batch vectorize all chunks
        chunk_texts = [c.text for c in chunks]
        vectors = await self._batch_vectorize(chunk_texts, retriever)

        if not vectors or len(vectors) != len(chunks):
            logger.error(
                "batch_vectorization_mismatch",
                expected=len(chunks),
                got=len(vectors) if vectors else 0,
            )
            # Fall back to individual vectorization
            vectors = await self._individual_vectorize(chunk_texts, retriever)

        # Step 2: Fan out parallel searches
        tasks = []
        for i, chunk in enumerate(chunks):
            chunk_vector = vectors[i] if i < len(vectors) else []
            task = asyncio.create_task(
                self._search_single_chunk(chunk, chunk_vector, retriever)
            )
            tasks.append(task)

        # Gather with per-chunk timeout
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Step 3: Collect results
        chunk_results: List[ChunkSearchResult] = []
        total_vector = 0
        total_graph = 0
        failed = 0

        for i, result in enumerate(raw_results):
            if isinstance(result, Exception):
                logger.error(
                    "chunk_search_failed",
                    chunk=chunks[i].text,
                    error=str(result),
                )
                chunk_results.append(ChunkSearchResult(
                    query_chunk=chunks[i],
                    error=str(result),
                ))
                failed += 1
            elif isinstance(result, ChunkSearchResult):
                chunk_results.append(result)
                total_vector += len(result.vector_results)
                total_graph += len(result.graph_results)
                if result.error:
                    failed += 1

        elapsed = (time.perf_counter() - start_time) * 1000

        result = ParallelSearchResult(
            chunk_results=chunk_results,
            total_vector_hits=total_vector,
            total_graph_hits=total_graph,
            total_time_ms=round(elapsed, 2),
            chunks_searched=len(chunks),
            chunks_failed=failed,
        )

        logger.info(
            "parallel_search_completed",
            chunks=len(chunks),
            vector_hits=total_vector,
            graph_hits=total_graph,
            failed=failed,
            time_ms=result.total_time_ms,
        )

        return result

    async def _search_single_chunk(
        self,
        chunk: QueryChunk,
        query_vector: List[float],
        retriever: IntelligentRetriever,
    ) -> ChunkSearchResult:
        """Search FalkorDB for a single query chunk with timeout."""
        start = time.perf_counter()

        try:
            result = await asyncio.wait_for(
                self._do_search(chunk, query_vector, retriever),
                timeout=self.per_chunk_timeout,
            )
            result.search_time_ms = round(
                (time.perf_counter() - start) * 1000, 2
            )
            return result
        except asyncio.TimeoutError:
            logger.warning("chunk_search_timeout", chunk=chunk.text)
            return ChunkSearchResult(
                query_chunk=chunk,
                error="timeout",
                search_time_ms=round(
                    (time.perf_counter() - start) * 1000, 2
                ),
            )

    async def _do_search(
        self,
        chunk: QueryChunk,
        query_vector: List[float],
        retriever: IntelligentRetriever,
    ) -> ChunkSearchResult:
        """Execute vector search + DFS for a single chunk."""
        vector_results: List[ChunkNode] = []
        graph_results: List[ChunkNode] = []

        if not query_vector:
            return ChunkSearchResult(
                query_chunk=chunk,
                vector_results=[],
                graph_results=[],
                error="empty_vector",
            )

        # Vector similarity search
        vector_results = await retriever.vector_search(
            query_vectors=query_vector,
            limit=self.vector_top_k,
        )

        # DFS traversal from top vector matches
        if vector_results:
            seed_ids = [r.chunk_id for r in vector_results[:3]]
            dfs_result = await retriever.dfs_traversal(
                start_chunk_ids=seed_ids,
                max_depth=self.dfs_depth,
                min_relevance=self.dfs_min_relevance,
                max_results=self.dfs_max_results,
            )
            graph_results = dfs_result.chunks

        return ChunkSearchResult(
            query_chunk=chunk,
            vector_results=vector_results,
            graph_results=graph_results,
        )

    async def _batch_vectorize(
        self, texts: List[str], retriever: IntelligentRetriever
    ) -> List[List[float]]:
        """
        Batch-vectorize multiple texts in a single HTTP call.
        Falls back to individual calls if batch endpoint is unavailable.
        """
        try:
            response = await retriever._http_client.post(
                f"{retriever.embeddings_service_url}/api/v1/generate-batch",
                json={"texts": texts},
                timeout=15.0,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("embeddings", [])
        except Exception as e:
            logger.warning(
                "batch_vectorize_failed_fallback_individual",
                error=str(e),
            )
            return await self._individual_vectorize(texts, retriever)

    async def _individual_vectorize(
        self, texts: List[str], retriever: IntelligentRetriever
    ) -> List[List[float]]:
        """Vectorize texts individually using existing vectorize_query."""
        tasks = [retriever.vectorize_query(t) for t in texts]
        return await asyncio.gather(*tasks)
