"""
Data Vent — Intelligent Retriever

Retrieval engine that routes queries to Graphify.
"""

from __future__ import annotations

import os
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Metadata keys extracted once for Graphify results
_GRAPHIFY_META_KEYS = ("id", "source_type", "source_id", "chunk_type", "content_path", "language")


class SearchResult:
    """Normalised retrieval result."""

    __slots__ = ("content", "score", "metadata", "source")

    def __init__(self, content: str, score: float, metadata: dict[str, Any], source: str = "graphify") -> None:
        self.content = content
        self.score = score
        self.metadata = metadata
        self.source = source

    def to_dict(self) -> dict[str, Any]:
        return {"content": self.content, "score": self.score, "metadata": self.metadata, "source": self.source}


class IntelligentRetriever:
    """
    Retrieval engine using GraphifyService.
    """

    __slots__ = ("_graphify",)

    def __init__(self, graphify_service: Any | None = None) -> None:
        self._graphify = graphify_service

    @property
    def active_backend(self) -> str:
        """Return the name of the currently active retrieval backend."""
        return "graphify"

    # ── Public API ────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        group_ids: list[str] | None = None,
        num_results: int = 10,
        center_node_uuid: str | None = None,
    ) -> list[SearchResult]:
        """
        Retrieve relevant knowledge from Graphify.
        """
        logger.info("retriever_search", query=query[:80], backend="graphify")

        if not self._graphify:
            logger.error("Graphify service is not configured")
            return []
            
        try:
            return await self._retrieve_graphify(query, group_ids, num_results, center_node_uuid)
        except Exception as exc:
            logger.warning("graphify_retrieval_failed", error=str(exc))
            return []

    async def retrieve_with_context(
        self,
        query: str,
        entity_uuid: str,
        group_ids: list[str] | None = None,
        num_results: int = 15,
    ) -> list[SearchResult]:
        """Retrieve knowledge centred around a specific entity node."""
        return await self.retrieve(query, group_ids, num_results, center_node_uuid=entity_uuid)

    # ── Private: Graphify ─────────────────────────────────────────────────────

    async def _retrieve_graphify(
        self, query: str, group_ids: list[str] | None, num_results: int, center_node_uuid: str | None,
    ) -> list[SearchResult]:
        raw = await self._graphify.search(
            query=query, group_ids=group_ids, num_results=num_results, center_node_uuid=center_node_uuid,
        )
        results = [
            SearchResult(
                content=r.get("chunk_text") or r.get("content", ""),
                score=float(r.get("score", 0.0)),
                metadata={k: r.get(k) for k in _GRAPHIFY_META_KEYS},
                source="graphify",
            )
            for r in raw
        ]
        logger.info("retriever_search_done", result_count=len(results), backend="graphify")
        return results
