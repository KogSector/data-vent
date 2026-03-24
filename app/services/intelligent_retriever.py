"""
Data Vent — Intelligent Retriever

Uses Graphiti for all semantic search and graph traversal operations.
FalkorDB with Redis protocol is used as the graph database backend.
"""

from __future__ import annotations

from typing import Any

import structlog

from app.services.graphiti_service import GraphitiService

logger = structlog.get_logger(__name__)


class SearchResult:
    """Normalised retrieval result."""

    def __init__(
        self,
        content: str,
        score: float,
        metadata: dict[str, Any],
        source: str = "graphiti",
    ) -> None:
        self.content = content
        self.score = score
        self.metadata = metadata
        self.source = source

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "source": self.source,
        }


class IntelligentRetriever:
    """
    Retrieval engine backed by Graphiti's hybrid vector + BM25 + graph search.

    Entity and relationship extraction is handled automatically by Graphiti
    during ingestion (in unified-processor). This class only queries.
    """

    def __init__(self, graphiti_service: GraphitiService) -> None:
        self._graphiti = graphiti_service

    async def retrieve(
        self,
        query: str,
        group_ids: list[str] | None = None,
        num_results: int = 10,
        center_node_uuid: str | None = None,
    ) -> list[SearchResult]:
        """
        Retrieve relevant knowledge from Graphiti.

        Combines vector similarity, BM25 lexical match, and graph-aware
        reranking in a single call (handled internally by graphiti-core).

        Args:
            query:            Natural language query
            group_ids:        Restrict search to specific graph groups / tenants
            num_results:      Maximum results to return
            center_node_uuid: If provided, rerank results around this graph node

        Returns:
            List of `SearchResult` ordered by relevance
        """
        logger.info("retriever_search", query=query[:80], group_ids=group_ids)

        raw_results = await self._graphiti.search(
            query=query,
            group_ids=group_ids,
            center_node_uuid=center_node_uuid,
            num_results=num_results,
        )

        results: list[SearchResult] = []
        for r in raw_results:
            # graphiti-core returns SearchResult-like objects; adapt to our model
            content = getattr(r, "fact", "") or getattr(r, "content", "")
            score = float(getattr(r, "score", 0.0) or 0.0)
            metadata: dict[str, Any] = {
                "uuid": getattr(r, "uuid", None),
                "name": getattr(r, "name", None),
                "created_at": str(getattr(r, "created_at", "")),
                "valid_at": str(getattr(r, "valid_at", "")),
            }
            results.append(SearchResult(content=content, score=score, metadata=metadata))

        logger.info("retriever_search_done", result_count=len(results))
        return results

    async def retrieve_with_context(
        self,
        query: str,
        entity_uuid: str,
        group_ids: list[str] | None = None,
        num_results: int = 15,
    ) -> list[SearchResult]:
        """
        Retrieve knowledge centred around a specific entity node.

        Useful for deep-dive queries where the caller already knows the
        relevant entity (e.g. a specific function or class).
        """
        return await self.retrieve(
            query=query,
            group_ids=group_ids,
            num_results=num_results,
            center_node_uuid=entity_uuid,
        )
